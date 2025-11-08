"""
Input Validation Utility
"""

import os
import re


def validate_file_path(path: str) -> tuple[bool, str]:
    """
    Validate file path
    
    Returns:
        (is_valid, error_message)
    """
    if not path:
        return False, "Path is empty."
    
    # Normalize path
    path = os.path.normpath(path)
    
    # Check dangerous patterns (Path Traversal prevention)
    if '..' in path:
        return False, "Parent directory access is not allowed."
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    return True, abs_path


def validate_filename(filename: str) -> tuple[bool, str]:
    """
    Validate filename
    
    Returns:
        (is_valid, error_message)
    """
    if not filename:
        return False, "Filename is empty."
    
    # Check dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        if char in filename:
            return False, f"Character '{char}' cannot be used in filename."
    
    # Check filename length
    if len(filename) > 255:
        return False, "Filename is too long (maximum 255 characters)."
    
    return True, filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename (convert to safe version)
    """
    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200] + ext
    
    return filename


def validate_csv_size(file_size_bytes: int, max_mb: int = 50) -> tuple[bool, str]:
    """
    Validate CSV file size
    
    Returns:
        (is_valid, error_message)
    """
    max_bytes = max_mb * 1024 * 1024
    
    if file_size_bytes > max_bytes:
        size_mb = file_size_bytes / (1024 * 1024)
        return False, f"File is too large ({size_mb:.1f}MB). Maximum {max_mb}MB allowed."
    
    return True, None


# Test
if __name__ == "__main__":
    print("✅ Validators Test")
    
    # Path validation
    valid, msg = validate_file_path("/home/user/data")
    print(f"Path validation: {valid} - {msg}")
    
    # Filename sanitization
    dirty = "sample<test>data?.csv"
    clean = sanitize_filename(dirty)
    print(f"Filename sanitization: '{dirty}' -> '{clean}'")
    
    print("✅ Module operational")