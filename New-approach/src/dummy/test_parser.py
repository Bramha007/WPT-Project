# File: test_parser.py (Temporary Script)

import os
import sys

# Add the source directory to the path so we can import the parser
# Assumes you are running this from the project root directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/dataio'))

from src.dataio.voc_parser import parse_voc

# --- CONFIGURATION (Change these paths!) ---
# You need to find one example XML file that contains a rectangle (W != H)
# and one example XML file that contains multiple objects.

# 1. Choose an XML file path that exists in your project.
# You can use the path structure from config_cls.py
# Example path for a test file on your Linux environment:
# TEST_XML_PATH = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled/annotations/00001.xml" 
# If running on Windows, use your Windows path:
# TEST_XML_PATH = r"E:\WPT-Project\Data\sized_rectangles_filled\annotations\1.xml"
TEST_XML_PATH = r"E:\WPT-Project\Data\sized_squares_filled\annotations\1.xml"
# --- END CONFIGURATION ---


def run_test():
    if not os.path.exists(TEST_XML_PATH):
        print(f"Error: XML file not found at {TEST_XML_PATH}. Please update TEST_XML_PATH.")
        return

    print(f"--- Testing voc_parser with: {TEST_XML_PATH} ---")
    try:
        results = parse_voc(TEST_XML_PATH)

        print("\n✅ Parser Output Success:")
        print(f"  Canvas Size: {results.get('width')} x {results.get('height')}")
        print(f"  Total Boxes Found: {len(results.get('boxes', []))}")
        
        # Check the crucial WxH labels
        wxh_labels = results.get('labels_wxh_str', [])
        
        print(f"  Total WxH Labels Found: {len(wxh_labels)}")
        
        # Print the list of boxes and their corresponding labels
        for i, (box, label) in enumerate(zip(results['boxes'], wxh_labels)):
            print(f"  Object {i+1}: BBox={box}, WxH Label='{label}'")

        # Check for consistency
        if len(results.get('boxes')) != len(wxh_labels):
            print("\n🚨 WARNING: Box count does NOT match WxH label count. Data pipeline will likely fail.")

    except Exception as e:
        print(f"\n❌ PARSER FAILED with exception: {e}")
        print("Please check the structure of your XML file and the error handling in parse_voc.")

if __name__ == "__main__":
    run_test()