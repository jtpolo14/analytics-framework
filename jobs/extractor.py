"""
Fetch National Hurricane Center XML data and upload to GCP bucket
"""
import requests
from google.cloud import storage
from datetime import datetime
import sys

def fetch_nhc_data(url: str):
    """
    Fetch XML data from NHC
    
    Args:
        url: URL of the NHC XML feed
        
    Returns:
        XML content as string
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise

def upload_to_gcs(bucket_name, data, blob_name=None):
    """
    Upload data to Google Cloud Storage bucket
    
    Args:
        bucket_name: Name of the GCS bucket
        data: String data to upload
        blob_name: Name for the file in GCS (optional, defaults to timestamped filename)
        
    Returns:
        Public URL of uploaded file
    """
    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Generate filename with timestamp if not provided
        if blob_name is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            blob_name = f"nhc_gtwo_{timestamp}.xml"
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type='application/xml')
        
        print(f"File uploaded successfully to gs://{bucket_name}/{blob_name}")
        return f"gs://{bucket_name}/{blob_name}"
        
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def main():
    """
    Main function to fetch NHC data and upload to GCS
    """
    # url of the NHC XML feed
    url = "https://www.nhc.noaa.gov/gtwo.xml"

    # Configuration
    BUCKET_NAME = "thomasanalytics-data1/lake/news/national_hurricane_center/graphical_tropical_weather_outlooks"  # Replace with your bucket name
    
    # Optional: specify a custom blob name, or leave as None for timestamped name
    BLOB_NAME = None  # e.g., "nhc/current_outlook.xml"
    
    print("Fetching NHC data...")
    xml_data = fetch_nhc_data(url)
    
    print(f"Data fetched successfully ({len(xml_data)} bytes)")
    
    print(f"Uploading to GCS bucket: {BUCKET_NAME}...")
    gcs_path = upload_to_gcs(BUCKET_NAME, xml_data, BLOB_NAME)
    
    print(f"✓ Complete! Data available at: {gcs_path}")

if __name__ == "__main__":
    main()