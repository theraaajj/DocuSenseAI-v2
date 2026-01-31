import os
import glob
from pathlib import Path

# Safety Blocklist- must not scanning entire OS or sensitive system folders
BLOCKED_DIRS = ["/", "/bin", "/Windows", "/System", "/usr", "/etc"]

class DiskScout:
    def __init__(self):
        self.allowed_paths = []

    def add_path(self, path_str):
        """
        Safely adds a folder to the allowlist.
        """
        path = Path(path_str).resolve()
        
        # validation checks
        if not path.exists():
            return False, "❌ Path does not exist."
        if not path.is_dir():
            return False, "❌ Path is not a directory."
        
        # check against blocklist
        for blocked in BLOCKED_DIRS:
            if str(path) == blocked:
                return False, "⛔ Security Alert: System root folders are blocked."
        
        # add to allowlist
        if path not in self.allowed_paths:
            self.allowed_paths.append(path)
            return True, f"✅ Access granted: {path.name}"
        return True, "Path already allowed."

    def scout_files(self, keyword):
        """
        Scans ALLOWED folders for filenames matching the keyword.
        Does NOT read file content. Fast & Private.
        """
        matches = []
        for folder in self.allowed_paths:
            # Simple case-insensitive filename match
            # look for keyword in the filename
            search_pattern = str(folder / "**" / f"*{keyword}*")
            
            # recursive search (depth is an imp thing in production)
            found_files = glob.glob(search_pattern, recursive=True)
            
            for f in found_files:
                p = Path(f)
                if p.is_file() and not p.name.startswith('.'): # Ignore hidden files
                    matches.append(p)
        
        # limit to top 10 matches - else context overflow
        return matches[:10]

# capable of extracting text from PDF, DOCX, XLSX, and Text files
    def read_file_lazy(self, file_path):
        path_str = str(file_path)
        ext = file_path.suffix.lower()

        try:
            # Excel
            if ext in [".xlsx", ".xls"]:
                xls = pd.read_excel(path_str, sheet_name=None)
                full_text = []
                for sheet_name, df in xls.items():
                    df = df.fillna("")
                    df.columns = df.columns.astype(str) # forced string headers
                    columns_list = ", ".join(list(df.columns))
                    
                    # prioritize the summary + sample for the 'Lazy Read'
                    # to avoid blowing up the context window with massive files.
                    full_text.append(f"""
                    SHEET: {sheet_name}
                    COLUMNS: {columns_list}
                    DATA SAMPLE:
                    {df.to_markdown(index=False)}
                    """)
                return "\n".join(full_text)

            elif ext == ".csv":
                df = pd.read_csv(path_str)
                df = df.fillna("")
                df.columns = df.columns.astype(str)
                columns_list = ", ".join(list(df.columns))
                
                return f"""
                FILE: {file_path.name}
                COLUMNS: {columns_list}
                DATA:
                {df.to_markdown(index=False)}
                """

            # Word
            elif ext == ".docx":
                return docx2txt.process(path_str)

            # PDF
            elif ext == ".pdf":
                return PyPDFLoader(path_str).load()
            
            # Text/Code
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()

        except Exception as e:
            return f"[Error reading file {file_path.name}: {e}]"