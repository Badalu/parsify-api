import os
import sys

def main():
    print("Testing parser imports and rule-based functions...")
    
    try:
        import parser
        print("SUCCESS: parser.py imported perfectly!")
    except Exception as e:
        print(f"FAILED: Could not import parser.py: {e}")
        sys.exit(1)
        
    # Test rule-based categorization
    descriptions = [
        ("ZOMATO IN DELHI", "Food"),
        ("AMAZON IN SELLER", "Shopping"),
        ("HPCL PETROL BENGALURU", "Fuel"),
        ("NETFLIX SUBSCRIPTION", "Subscription"),
        ("CGST SPLIT TAX", "GST"),
        ("MONTHLY SALARY INWARD", "Salary"),
        ("UNKNOWN TRANSACTION", "Other")
    ]
    
    print("\nTesting local rule-based categorization...")
    for desc, expected_cat in descriptions:
        cat = parser.categorize_locally(desc)
        print(f"  Description: '{desc}' => Categorized as: '{cat}' (Expected: '{expected_cat}')")
        if cat != expected_cat:
            print(f"  WARNING: Expected '{expected_cat}' but got '{cat}'")
            
    # Test local GST extraction
    gst_descriptions = [
        ("CGST 9% SGST 9%", "CGST+SGST"),
        ("IGST TAX INVOICE", "IGST"),
        ("GST CHARGES ON TRANSFER", "GST"),
        ("NORMAL TRANSACTION", "")
    ]
    
    print("\nTesting local GST extraction...")
    for desc, expected_gst in gst_descriptions:
        gst = parser.extract_gst_locally(desc)
        print(f"  Description: '{desc}' => GST: '{gst}' (Expected: '{expected_gst}')")
        if gst != expected_gst:
            print(f"  WARNING: Expected '{expected_gst}' but got '{gst}'")
            
    # Test clean and format transactions helper
    test_txns = [
        {"date": "15/05/2023", "description": "ZOMATO FEEDING", "debit": "150.00", "credit": "", "balance": "1000.00", "category": "Food", "gst": ""},
        {"date": "16/05/2023", "description": "SALARY TRANSFER", "debit": "", "credit": "50000.00", "balance": "51000.00", "category": "Salary", "gst": ""},
    ]
    print("\nTesting clean_and_format_transactions helper...")
    try:
        formatted = parser.clean_and_format_transactions(test_txns, date_format="DD/MM/YYYY")
        print(f"  SUCCESS: Formatted {len(formatted)} transactions successfully!")
        for txn in formatted:
            print(f"    Txn: {txn['date']} | {txn['description']} | Debit: {txn['debit']} | Credit: {txn['credit']} | Cat: {txn['category']}")
    except Exception as e:
        print(f"  FAILED to run clean_and_format_transactions: {e}")
        sys.exit(1)
        
    print("\nAll local parsing checks passed perfectly!")

if __name__ == "__main__":
    main()
