import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import parser

class TestParserAdvanced(unittest.TestCase):

    def test_date_detection(self):
        valid_dates = [
            "12/05/2023", "12-05-23", "12.05.2023",
            "12-May-2023", "12-September-23",
            "12 May 2023", "12 May 23", "12May2023", "12May23",
            "May 12, 2023", "May 12 23",
            "2023-05-12",
            "12/05/23",
            "12 / 05 / 2023",
            "12052023",
            "12-May", "12 May"
        ]
        
        invalid_dates = [
            "123/45/2023", "not-a-date", "", "100.00", "Rs. 500"
        ]
        
        for d in valid_dates:
            with self.subTest(date=d):
                self.assertTrue(parser.is_valid_date(d), f"Should be valid: {d}")
                
        for d in invalid_dates:
            with self.subTest(date=d):
                self.assertFalse(parser.is_valid_date(d), f"Should be invalid: {d}")

    def test_find_header_mapping(self):
        # Test standard format
        row1 = ["Date", "Particulars", "Chq/Ref No.", "Withdrawal (Dr)", "Deposit (Cr)", "Balance"]
        map1 = parser.find_header_mapping(row1)
        self.assertIn("date", map1)
        self.assertIn("description", map1)
        self.assertIn("debit", map1)
        self.assertIn("credit", map1)
        self.assertIn("balance", map1)
        
        # Test borderless/single-amount format
        row2 = ["Post Date", "Value Date", "Description", "Amount", "Dr/Cr", "Balance"]
        map2 = parser.find_header_mapping(row2)
        self.assertIn("date", map2)
        self.assertIn("value_date", map2)
        self.assertIn("description", map2)
        self.assertIn("amount", map2)
        self.assertIn("dr_cr", map2)
        self.assertIn("balance", map2)

    def test_amount_detection_and_cleaning(self):
        # Test helper check
        self.assertTrue(parser._is_amount_str("1,00,000.00"))
        self.assertTrue(parser._is_amount_str("(500.00)"))
        self.assertTrue(parser._is_amount_str("1,234.56 Dr"))
        self.assertTrue(parser._is_amount_str("₹ 5,000.00"))
        self.assertTrue(parser._is_amount_str("10, 000.00"))
        self.assertFalse(parser._is_amount_str("NoAmount"))

        # Test clean_and_format_transactions helper
        txns = [
            {"date": "12/05/2023", "description": "Zomato", "debit": "1,00,000.00", "credit": "", "balance": "2,00,000.00"},
            {"date": "13/05/2023", "description": "Salary", "debit": "", "credit": "₹ 50,000.50 Cr", "balance": "2,50,000.50"},
            {"date": "14/05/2023", "description": "Refund", "debit": "(250.00)", "credit": "", "balance": "2,49,750.50"},
            {"date": "15/05/2023", "description": "Cash Withdrawal", "debit": " 10, 000.00 ", "credit": "", "balance": "2,39,750.50"},
        ]
        
        cleaned = parser.clean_and_format_transactions(txns)
        self.assertEqual(cleaned[0]["debit"], "100000.00")
        self.assertEqual(cleaned[0]["balance"], "200000.00")
        self.assertEqual(cleaned[1]["credit"], "50000.50")
        self.assertEqual(cleaned[2]["debit"], "250.00")
        self.assertEqual(cleaned[3]["debit"], "10000.00")

    def test_parse_line_by_line(self):
        # Simulate a printed statement layout
        text = """
        HDFC BANK STATEMENT
        Date        Particulars                  Chq No.   Debit      Credit     Balance
        12/05/2023  UPI-ZOMATO-12345             000000    150.00                10000.00
        13/05/2023  SALARY INWARD                          50000.00   60000.00
                    REMAINING DESC
        14/05/2023  INTEREST RECEIVED                                 250.00     60250.00
        TOTALS      -                            -         150.00     50250.00
        """
        
        txns = parser.parse_line_by_line(text)
        self.assertEqual(len(txns), 3)
        self.assertEqual(txns[0]["date"], "12/05/2023")
        self.assertEqual(txns[0]["debit"], "150.00")
        self.assertEqual(txns[0]["balance"], "10000.00")
        
        # Test description continuation line
        self.assertEqual(txns[1]["date"], "13/05/2023")
        self.assertIn("REMAINING DESC", txns[1]["description"])
        self.assertEqual(txns[1]["debit"], "50000.00")
        self.assertEqual(txns[1]["credit"], "60000.00")
        
        self.assertEqual(txns[2]["date"], "14/05/2023")
        self.assertEqual(txns[2]["credit"], "250.00")

    @patch("parser.PdfReader")
    @patch("pdfplumber.open")
    def test_parse_pdf_natively_mocked(self, mock_open, mock_pdf_reader):
        # Mock PdfReader
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]
        mock_pdf_reader.return_value = mock_reader

        # Set up mock pdf structure
        mock_pdf = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_pdf
        
        mock_page = MagicMock()
        mock_pdf.pages = [mock_page]
        
        # Mock table with headers and data
        mock_table = [
            ["Date", "Particulars", "Debit", "Credit", "Balance"],
            ["12-05-2023", "UPI Transfer", "500.00", "", "9500.00"],
            ["13-05-2023", "Salary", "", "50000.00", "59500.00"],
            ["", "Continuation Line of UPI Transfer", "", "", ""],
        ]
        
        # We want extract_tables to return our mock_table
        mock_page.extract_tables.return_value = [mock_table]
        
        txns = parser.parse_pdf_natively("dummy.pdf")
        self.assertEqual(len(txns), 2)
        self.assertEqual(txns[0]["date"], "12-05-2023")
        self.assertEqual(txns[0]["debit"], "500.00")
        self.assertEqual(txns[1]["date"], "13-05-2023")
        self.assertEqual(txns[1]["credit"], "50000.00")

if __name__ == "__main__":
    unittest.main()
