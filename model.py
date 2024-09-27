from fastapi.responses import StreamingResponse
# from pdf2image import convert_from_path
import io
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import HttpResponseError
import pandas as pd
import os
from fastapi import File,UploadFile, HTTPException
import requests 
import fitz
from PIL import Image

# Azure Blob Storage connection string
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aisa0101;AccountKey=rISVuOQPHaSssHHv/dQsDSKBrywYnk6bNuXuutl4n+ILZNXx/CViS50NUn485kzsRxd5sfiVSsMi+AStga0t0g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "bankimage"

PREDICTION_KEY = '20c0b2ad608c42ca87a79ce5759b0d4e'
PREDICTION_URL = "https://bankclass-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/0fcd9dd4-d8eb-4292-bb15-b3c4819fcc18/classify/iterations/Iteration9/image"
 
# Initialize the client
endpoint = "https://eastus.api.cognitive.microsoft.com/"
key = "26cfaa6c7c314e9a8ad7a68587ca3ce9"


def convert_pdf_to_image(pdf_path, first_page=1, last_page=1):
    """Convert specified pages of a PDF file to images."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(first_page - 1, last_page):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return None
    
def process_pdf(pdf_path):
    try:
        # Convert the first page of the PDF to an image
        images = convert_pdf_to_image(pdf_path, first_page=1, last_page=1)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None
 
    if images:
        # Convert image to bytes in a supported format
        image_stream = io.BytesIO()
        try:
            images[0].save(image_stream, format='PNG')  # Use PNG format
            image_stream.seek(0)  # Reset stream position
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
       
        return image_stream
    else:
        print("No pages found in the PDF.")
        return None
 
#------------------------------------------------------------------------------------------
 
# Global variable to store the predicted class
predicted_class = None
 
def get_prediction_from_stream(image_stream):
    headers = {
        'Prediction-Key': PREDICTION_KEY,
        'Content-Type': 'application/pdf'
    }
    try:
        response = requests.post(PREDICTION_URL, headers=headers, data=image_stream)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")
 
def predict_class(image_stream):
    global predicted_class
 
    # Reset the stream position before sending for prediction
    image_stream.seek(0)
 
    try:
        prediction = get_prediction_from_stream(image_stream)
 
        if 'predictions' not in prediction:
            raise ValueError("Invalid response format: 'predictions' key missing")
 
        # Get the top prediction
        top_prediction = max(prediction['predictions'], key=lambda x: x['probability'])
        predicted_class = top_prediction['tagName']
        probability = top_prediction['probability']
 
        print(f"Predicted Class: {predicted_class}")
        print(f"Probability: {probability:.2f}")
 
    except Exception as e:
        print(f"Error in prediction: {e}")
 
 
ALLOWED_EXTENSIONS = {'pdf'}
 
 
def allowed_file(filename: str):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
async def upload_bank_file(file: UploadFile = File(...)):
    """Handle file uploads and processing."""
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are allowed.")
 
    filename = file.filename
    # file_path = os.path.join("./", filename)
 
    # Save the file to the upload folder
    try:
        with open(filename, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
   
    try:
        # Process the PDF to a stream
        image_stream = process_pdf(filename)  # Call your process_pdf function
 
        if image_stream:
            # Perform classification
            result = predict_class(image_stream)  # Assuming it returns a result object
 
            # Analyze the document further
            analyze_document(image_stream,predicted_class)  # Perform additional analysis
 
            # Analyze the PDF for transaction tables
            analyze_pdf(filename,None,None,None)  # Analyze the PDF for tables
 
            return {"message": "PDF processed successfully."}
 
        else:
            raise HTTPException(status_code=500, detail="Error processing PDF.")
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
 
account_name_str = None
account_no_str = None
address_str = None
branch_str = None
 
def analyze_document(image_stream, predicted_class):
    global account_name_str, account_no_str, address_str, branch_str
   
    # Initialize the model ID based on predicted class
    model_id = predicted_class
   
    if model_id and image_stream:
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
 
        try:
            # Reset the stream position before sending for analysis
            image_stream.seek(0)
 
            # Analyze the document using the in-memory image
            poller = document_analysis_client.begin_analyze_document(model_id=model_id, document=image_stream)
            result = poller.result()
 
            if not result.documents:
                print("No documents found in the analysis result.")
                return None, None, None, None
 
            for idx, document in enumerate(result.documents):
                print(f"--------Analyzing document #{idx + 1}--------")
                print(f"Document has type {document.doc_type}")
                print(f"Document has confidence {document.confidence}")
                print(f"Document was analyzed by model with ID {result.model_id}")
 
                fields_data = {"Field Name": [], "Value": []}
 
                for name, field in document.fields.items():
                    field_value = field.value if field.value else field.content
                    fields_data["Field Name"].append(name)
                    fields_data["Value"].append(field_value)
 
                # Create DataFrame for fields data
                fields_df = pd.DataFrame(fields_data)
 
                # Extract specific fields of interest with improved matching
                account_name = fields_df.loc[
                    fields_df["Field Name"].str.contains("account name|name|holder", case=False, na=False), "Value"
                ]
                account_no = fields_df.loc[
                    fields_df["Field Name"].str.contains("account", case=False, na=False) &
                    fields_df["Field Name"].str.contains("no", case=False, na=False), "Value"
                ]
                address = fields_df.loc[
                    fields_df["Field Name"].str.contains("address", case=False, na=False), "Value"
                ]
                branch = fields_df.loc[
                    fields_df["Field Name"].str.contains("branch", case=False, na=False), "Value"
                ]
 
                account_name_str = " ".join(account_name.tolist()) if not account_name.empty else "NotFound"
                account_no_str = account_no.iloc[0] if not account_no.empty else "NotFound"
                address_str = address.iloc[0] if not address.empty else "NotFound"
                branch_str = branch.iloc[0] if not branch.empty else "NotFound"
 
                print(f"Account Name: {account_name_str}")
                print(f"Account No: {account_no_str}")
                print(f"Address: {address_str}")
                print(f"Branch: {branch_str}")
 
                # Concatenate and print the final string without commas or spaces
                global concatenated_info
                concatenated_info = f"{account_name_str}{account_no_str}"
                print(f"Concatenated Info: {concatenated_info}")
 
                return account_name_str, account_no_str, address_str, branch_str
 
        except HttpResponseError as e:
            print(f"Error analyzing document: {e}")
            return None, None, None, None
 
# Function to retrieve extracted values without parameters
def get_extracted_values():
    global account_name_str, account_no_str, address_str, branch_str
    return account_name_str, account_no_str, address_str, branch_str
 
 
 
 
#--------------------------------------------------------------------------
 
document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)
 
 
def make_unique_columns(columns):
    """Ensure column names are unique."""
    seen = set()
    unique_columns = []
    for col in columns:
        new_col = col
        count = 1
        while new_col in seen:
            new_col = f"{col}_{count}"
            count += 1
        seen.add(new_col)
        unique_columns.append(new_col)
    return unique_columns
 
concatenated_info = f"{account_name_str}{account_no_str}"
folder_name = f"{concatenated_info}"
 
 
def analyze_pdf(pdf_path, account_name_str, account_no_str, branch_str):
    """Analyze a PDF using Azure Form Recognizer, extract tables with 'balance' column, and drop duplicate headers."""
    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return []
 
    keyword_to_detect = 'balance'
    first_page_processed = False  # Flag to track if the first page has been processed
    headers_set = False  # To indicate if headers have been set
    column_names = []  # Store column names from the first page
 
    try:
        with open(pdf_path, "rb") as pdf_stream:
            poller = document_analysis_client.begin_analyze_document("prebuilt-document", pdf_stream)
            result = poller.result()
 
        all_tables = []
 
        # Check if the result contains tables
        if result.tables:
            print(f"Found {len(result.tables)} tables in {pdf_path}")
            for table_idx, table in enumerate(result.tables):
                print(f"Processing Table #{table_idx} from {pdf_path}:")
 
                # Create an empty matrix for the table
                matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
 
                # Populate the matrix with cell content
                for cell in table.cells:
                    row_index = cell.row_index
                    column_index = cell.column_index
                    matrix[row_index][column_index] = cell.content.strip()  # Clean whitespace
 
                # Convert the matrix to a DataFrame
                df = pd.DataFrame(matrix)
 
                # Set headers from the first page and check for duplicates on subsequent pages
                if not first_page_processed:
                    # Process the first page
                    if not headers_set and not df.empty:
                        # Check if the first row contains 'balance'
                        if any(keyword_to_detect in str(col).lower() for col in df.iloc[0]):
                            # Set column names from the first row
                            column_names = make_unique_columns(df.iloc[0])
                            df = df[1:]  # Drop header row from the data
                            headers_set = True
                            first_page_processed = True  # Mark that the first page has been processed
 
                            # Ensure the number of columns matches
                            if len(df.columns) == len(column_names):
                                df.columns = column_names
                            else:
                                df.columns = make_unique_columns([f"col_{i}" for i in range(len(df.columns))])
 
                            # Append the cleaned DataFrame to the list
                            all_tables.append(df)
 
                else:
                    # For subsequent pages, check if any row matches the header and drop it
                    def is_duplicate_header(row):
                        return all(col in column_names for col in row)
 
                    df = df[~df.apply(is_duplicate_header, axis=1)]  # Drop rows that are duplicate headers
 
                    # Set column names using the initial column_names
                    if len(df.columns) == len(column_names):
                        df.columns = column_names
                    else:
                        df.columns = make_unique_columns([f"col_{i}" for i in range(len(df.columns))])
 
                    # Append the DataFrame only if it contains 'balance' column
                    if any(keyword_to_detect in str(col).lower() for col in df.columns):
                        all_tables.append(df)
 
        # Combine all collected DataFrames
        if all_tables:
            combined_df = pd.concat(all_tables, ignore_index=True)
 
            # Saving the final DataFrame
            file_name = "Bank_Statement.csv"
            global folder_name
            folder_name = f"{concatenated_info}"
            save_dataframe_to_csv_and_blob(combined_df, folder_name, file_name)
 
            return [combined_df]
 
        return []
 
    except HttpResponseError as e:
        print("Error analyzing document:", e)
        return []
 
 
# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
 
def upload_to_blob(blob_name:str, file_content:str):
    """Upload a file to Azure Blob Storage."""
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        content_settings = ContentSettings(content_type='text/csv')
        blob_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)
        print(f"File '{blob_name}' uploaded to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading file to Azure Blob Storage: {e}")
 
def save_dataframe_to_csv_and_blob(df, folder_name, file_name):
    """Save a DataFrame to a CSV file and upload it to Azure Blob Storage in a specified folder."""
    # Save DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Create the full blob name including the folder
    full_blob_name = f"{folder_name}/{file_name}"

    # Upload the CSV file content to Azure Blob Storage
    upload_to_blob(full_blob_name, csv_buffer.getvalue())



def upload_pdf_to_blob(blob_name, file_content):
    """Upload a file to Azure Blob Storage."""
    blob_name = f"{folder_name}/{blob_name}.pdf"
    try:
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        content_settings = ContentSettings(content_type='application/pdf')

        blob_client.upload_blob(file_content, overwrite=True, content_settings=content_settings)

        print(f"File '{blob_name}' uploaded to Azure Blob Storage.")
    except Exception as e:
        print(f"Error uploading file to Azure Blob Storage: {e}")


# def download_csv(blob_name: str, file_extension: str='csv'):
#     """Download a file from Azure Blob Storage based on file extension."""
#     blob_name = f"{folder_name}/{blob_name}.{file_extension}"
#     try:
#         blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
#         download_stream = blob_client.download_blob()
#         return StreamingResponse(
#             iter(download_stream.chunks()),
#             media_type="text/csv",
#             headers={"Content-Disposition": f'attachment; filename="{blob_name}"'}
#         )
        
#     except Exception as e:
#         if file_extension.lower() == "pdf":
#             raise HTTPException(status_code=500, detail=f"Error downloading PDF file: {str(e)}")
#         elif file_extension.lower() == "csv":
#             raise HTTPException(status_code=500, detail=f"Error downloading CSV file: {str(e)}")
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file extension.")


def view_pdf(blob_name: str, file_extension: str):
    """Stream a file from Azure Blob Storage based on file extension without saving it locally."""
    blob_name = f"{folder_name}/{blob_name}.pdf"
    try:
        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        # Download the file as a stream
        download_stream = blob_client.download_blob()

        # Return the download stream for immediate use (viewing in browser)
        return StreamingResponse(
            iter(download_stream.chunks()),  # Stream the file content
            media_type=f"application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{blob_name}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading {file_extension.upper()} file: {str(e)}")


def view_csv(blob_name: str, file_extension: str):
    """Stream a file from Azure Blob Storage based on file extension without saving it locally."""
    blob_name = f"{folder_name}/{blob_name}.csv"
    try:
        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        # Download the file as a stream
        download_stream = blob_client.download_blob()

        # Return the download stream for immediate use (viewing in browser)
        return StreamingResponse(
            iter(download_stream.chunks()),  # Stream the file content
            media_type=f"application/csv",
            headers={
                "Content-Disposition": f'inline; filename="{blob_name}"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading {file_extension.upper()} file: {str(e)}")