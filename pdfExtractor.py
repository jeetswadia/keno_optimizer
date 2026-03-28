"""
keno_pdf_extractor.py
=====================
Extracts Massachusetts Keno draw data from PDF files
and outputs a clean CSV ready for the prediction agent.

Install requirements:
    pip install pdfplumber pandas
"""

import re
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system(f"{sys.executable} -m pip install pdfplumber")
    import pdfplumber

import pandas as pd


@dataclass
class KenoDraw:
    """Represents a single Keno draw."""
    game_id: str
    numbers: List[int]
    bonus: str  # "No Bonus", "2X", "3X", "4X", "5X", "10X"
    date: str = ""


class KenoPDFExtractor:
    """
    Extracts Keno results from Massachusetts Lottery PDF format.
    
    The PDF has this structure per draw:
        GAME_ID
        n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15
        n16 n17 n18 n19 n20
        [No] Bonus[:] [multiplier]
    
    Numbers are always 20 per draw, range 1-80.
    """
    
    VALID_RANGE = range(1, 81)
    NUMBERS_PER_DRAW = 20
    GAME_ID_PATTERN = re.compile(r'^29\d{5}$')  # MA Keno IDs like 2987605
    
    def __init__(self):
        self.draws: List[KenoDraw] = []
        self.date: str = ""
    
    def extract_from_pdf(self, pdf_path: str) -> List[KenoDraw]:
        """
        Extract all Keno draws from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of KenoDraw objects, ordered oldest-first
        """
        print(f"📄 Opening PDF: {pdf_path}")
        
        all_text_lines = []
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   Pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    lines = text.strip().split('\n')
                    all_text_lines.extend(lines)
        
        # Extract date from header
        self._extract_date(all_text_lines)
        
        # Parse draws
        self.draws = self._parse_lines(all_text_lines)
        
        # Sort by game_id ascending (oldest first)
        self.draws.sort(key=lambda d: int(d.game_id))
        
        print(f"✅ Extracted {len(self.draws)} draws")
        if self.draws:
            print(f"   Date: {self.date}")
            print(f"   Game IDs: {self.draws[0].game_id} → {self.draws[-1].game_id}")
            print(f"   Sample draw: {self.draws[-1].numbers}")
        
        # Validate
        self._validate()
        
        return self.draws
    
    def _extract_date(self, lines: List[str]):
        """Extract the draw date from header lines."""
        for line in lines[:10]:
            # Look for date pattern like 2026-03-27
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
            if date_match:
                self.date = date_match.group(1)
                return
            # Also try MM/DD/YY format
            date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', line)
            if date_match:
                self.date = date_match.group(1)
                return
    
    def _parse_lines(self, lines: List[str]) -> List[KenoDraw]:
        """
        Parse text lines into KenoDraw objects.
        
        Strategy: Find game IDs, then collect numbers that follow,
        and capture bonus information.
        """
        draws = []
        
        # Clean and tokenize all lines
        tokens = []
        for line in lines:
            # Skip header/footer lines
            if any(skip in line.lower() for skip in [
                'game name', 'game results', 'past results',
                'masslottery.com', 'https://', 'date:'
            ]):
                continue
            
            # Split line into individual tokens
            parts = line.strip().split()
            tokens.extend(parts)
        
        # Now walk through tokens, finding game IDs and collecting numbers
        i = 0
        current_game_id = None
        current_numbers = []
        current_bonus = "No Bonus"
        
        while i < len(tokens):
            token = tokens[i].strip().rstrip(',').rstrip('.')
            
            # Check if this is a game ID
            if self.GAME_ID_PATTERN.match(token):
                # Save previous draw if complete
                if current_game_id and len(current_numbers) == self.NUMBERS_PER_DRAW:
                    draws.append(KenoDraw(
                        game_id=current_game_id,
                        numbers=sorted(current_numbers),
                        bonus=current_bonus,
                        date=self.date
                    ))
                elif current_game_id and len(current_numbers) > 0:
                    # Partial draw - might happen at page boundaries
                    if len(current_numbers) == self.NUMBERS_PER_DRAW:
                        draws.append(KenoDraw(
                            game_id=current_game_id,
                            numbers=sorted(current_numbers),
                            bonus=current_bonus,
                            date=self.date
                        ))
                
                # Start new draw
                current_game_id = token
                current_numbers = []
                current_bonus = "No Bonus"
                i += 1
                continue
            
            # Check if this is a number in valid range
            try:
                num = int(token)
                if num in self.VALID_RANGE and len(current_numbers) < self.NUMBERS_PER_DRAW:
                    current_numbers.append(num)
                    i += 1
                    continue
            except ValueError:
                pass
            
            # Check for bonus info
            if token.lower() in ['bonus', 'bonus:']:
                i += 1
                continue
            if token.lower() == 'no':
                i += 1
                continue
            if token.upper() in ['2X', '3X', '4X', '5X', '10X']:
                current_bonus = token.upper()
                i += 1
                continue
            
            # Page numbers like "1/32", "2/32"
            if '/' in token and all(p.isdigit() for p in token.split('/')):
                i += 1
                continue
            
            i += 1
        
        # Don't forget the last draw
        if current_game_id and len(current_numbers) == self.NUMBERS_PER_DRAW:
            draws.append(KenoDraw(
                game_id=current_game_id,
                numbers=sorted(current_numbers),
                bonus=current_bonus,
                date=self.date
            ))
        
        return draws
    
    def _validate(self):
        """Validate extracted data."""
        issues = 0
        for draw in self.draws:
            if len(draw.numbers) != self.NUMBERS_PER_DRAW:
                print(f"   ⚠️  Draw {draw.game_id}: has {len(draw.numbers)} numbers (expected {self.NUMBERS_PER_DRAW})")
                issues += 1
            
            if len(set(draw.numbers)) != len(draw.numbers):
                print(f"   ⚠️  Draw {draw.game_id}: has duplicate numbers")
                issues += 1
            
            for n in draw.numbers:
                if n not in self.VALID_RANGE:
                    print(f"   ⚠️  Draw {draw.game_id}: invalid number {n}")
                    issues += 1
        
        if issues == 0:
            print("   ✅ All draws validated successfully")
        else:
            print(f"   ⚠️  {issues} validation issues found")
    
    def to_csv(self, output_path: str):
        """
        Save extracted draws to CSV.
        
        CSV format:
            game_id, date, n1, n2, ..., n20, bonus
        """
        if not self.draws:
            print("No draws to save!")
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            num_cols = [f'n{i+1}' for i in range(self.NUMBERS_PER_DRAW)]
            writer.writerow(['game_id', 'date'] + num_cols + ['bonus'])
            
            # Data rows
            for draw in self.draws:
                row = [draw.game_id, draw.date] + draw.numbers + [draw.bonus]
                writer.writerow(row)
        
        print(f"💾 Saved {len(self.draws)} draws to: {output_path}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = []
        for draw in self.draws:
            record = {
                'game_id': draw.game_id,
                'date': draw.date,
                'bonus': draw.bonus,
            }
            for i, num in enumerate(draw.numbers):
                record[f'n{i+1}'] = num
            records.append(record)
        
        return pd.DataFrame(records)
    
    def get_numbers_only(self) -> List[List[int]]:
        """Get just the number lists (for the prediction agent)."""
        return [draw.numbers for draw in self.draws]


class MultiPDFExtractor:
    """
    Handle multiple PDF files to build a larger dataset.
    Also supports scraping directly from the website URL pattern.
    """
    
    def __init__(self):
        self.all_draws: List[KenoDraw] = []
        self.extractor = KenoPDFExtractor()
    
    def extract_folder(self, folder_path: str, pattern: str = "*.pdf") -> List[KenoDraw]:
        """Extract from all PDFs in a folder."""
        folder = Path(folder_path)
        pdf_files = sorted(folder.glob(pattern))
        
        print(f"📁 Found {len(pdf_files)} PDF files in {folder_path}")
        
        for pdf_file in pdf_files:
            try:
                ext = KenoPDFExtractor()
                draws = ext.extract_from_pdf(str(pdf_file))
                self.all_draws.extend(draws)
            except Exception as e:
                print(f"   ❌ Error processing {pdf_file}: {e}")
        
        # Deduplicate by game_id
        seen = set()
        unique_draws = []
        for draw in self.all_draws:
            if draw.game_id not in seen:
                seen.add(draw.game_id)
                unique_draws.append(draw)
        
        self.all_draws = sorted(unique_draws, key=lambda d: int(d.game_id))
        print(f"\n📊 Total unique draws: {len(self.all_draws)}")
        
        return self.all_draws
    
    def to_csv(self, output_path: str):
        """Save all draws to CSV."""
        ext = KenoPDFExtractor()
        ext.draws = self.all_draws
        ext.to_csv(output_path)


# ─────────────────────────────────────────────────────────
# ALTERNATIVE: Extract from raw text (if PDF parsing fails)
# ─────────────────────────────────────────────────────────

def extract_from_raw_text(text: str, date: str = "") -> List[KenoDraw]:
    """
    Fallback parser for raw copy-pasted text from the PDF.
    
    Just paste the entire PDF text content as a string.
    """
    extractor = KenoPDFExtractor()
    extractor.date = date
    lines = text.strip().split('\n')
    draws = extractor._parse_lines(lines)
    draws.sort(key=lambda d: int(d.game_id))
    return draws


# ─────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────

def extract_and_save(pdf_path: str, csv_output: str = None):
    """
    One-liner to extract PDF and save CSV.
    
    Usage:
        python keno_pdf_extractor.py "Past Results 2026-03-27.pdf"
    """
    if csv_output is None:
        csv_output = pdf_path.replace('.pdf', '.csv')
    
    extractor = KenoPDFExtractor()
    draws = extractor.extract_from_pdf(pdf_path)
    extractor.to_csv(csv_output)
    
    # Also show quick stats
    df = extractor.to_dataframe()
    print(f"\n📈 Quick Stats:")
    print(f"   Total draws: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Number frequency
    all_nums = []
    for draw in draws:
        all_nums.extend(draw.numbers)
    from collections import Counter
    freq = Counter(all_nums)
    print(f"   Most common: {freq.most_common(5)}")
    print(f"   Least common: {freq.most_common()[-5:]}")
    
    return extractor


if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_and_save(sys.argv[1])
    else:
        print("Usage: python keno_pdf_extractor.py <pdf_file>")
        print("   or: python keno_pdf_extractor.py <folder_of_pdfs>")