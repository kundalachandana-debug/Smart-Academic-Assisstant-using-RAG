import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import PDF_DIR, VECTORSTORE_DIR

MAX_FILE_SIZE_MB   = 50
MAX_BATCH_SIZE     = 20
CONCURRENCY_LIMIT  = 4

ALLOWED_EXTENSIONS = {".pdf", ".ppt", ".pptx", ".docx", ".doc", ".txt", ".csv"}

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/csv",
}

MAGIC_BYTES = {
    ".pdf":  b"%PDF",
    ".docx": b"PK\x03\x04",
    ".pptx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
}


def validate_file(uploaded_file):
    name = uploaded_file.name
    ext  = os.path.splitext(name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False, f"'{name}': unsupported file type '{ext}'"

    content = uploaded_file.read()
    uploaded_file.seek(0)
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"'{name}': file size {size_mb:.1f} MB exceeds {MAX_FILE_SIZE_MB} MB limit"

    if len(content) == 0:
        return False, f"'{name}': file is empty"

    if ext in MAGIC_BYTES:
        expected_magic = MAGIC_BYTES[ext]
        if not content[:len(expected_magic)] == expected_magic:
            return False, f"'{name}': file appears corrupted or is not a valid {ext.upper()} file"

    return True, ""


def validate_batch(uploaded_files):
    if not uploaded_files:
        return [], ["No files provided"]

    if len(uploaded_files) > MAX_BATCH_SIZE:
        return [], [f"Batch size {len(uploaded_files)} exceeds maximum of {MAX_BATCH_SIZE} files"]

    valid_files = []
    errors      = []

    for f in uploaded_files:
        is_valid, error_msg = validate_file(f)
        if is_valid:
            valid_files.append(f)
        else:
            errors.append(error_msg)

    return valid_files, errors


def _save_single_file(uploaded_file):
    name = uploaded_file.name
    try:
        os.makedirs(PDF_DIR, exist_ok=True)
        dest_path = os.path.join(PDF_DIR, name)
        uploaded_file.seek(0)
        with open(dest_path, "wb") as out:
            out.write(uploaded_file.read())
        return {"file": name, "path": dest_path, "status": "saved", "error": None}
    except Exception as e:
        return {"file": name, "path": None, "status": "error", "error": str(e)}


def save_uploaded_pdf(uploaded_file):
    result = _save_single_file(uploaded_file)
    if result["status"] == "error":
        raise IOError(f"Could not save '{result['file']}': {result['error']}")
    return result["path"]


def save_uploaded_files_batch(uploaded_files):
    saved_paths = []
    errors      = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        futures = {executor.submit(_save_single_file, f): f.name for f in uploaded_files}
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "saved":
                saved_paths.append(result["path"])
            else:
                errors.append(f"'{result['file']}': {result['error']}")

    return saved_paths, errors


def convert_to_pdf_if_needed(file_path):
    if not file_path.lower().endswith((".ppt", ".pptx")):
        return file_path
    try:
        import subprocess
        out_dir = os.path.dirname(file_path)
        subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf", file_path, "--outdir", out_dir],
            check=True, capture_output=True, timeout=60,
        )
        pdf_path = os.path.splitext(file_path)[0] + ".pdf"
        if os.path.exists(pdf_path):
            os.remove(file_path)
            return pdf_path
    except Exception as e:
        print(f"[convert_to_pdf] Warning — could not convert '{file_path}': {e}")
    return file_path


def convert_batch_to_pdf(file_paths):
    with ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        results = list(executor.map(convert_to_pdf_if_needed, file_paths))
    return results


def rebuild_vectorstore():
    from ingest import load_and_split, build_vectorstore
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
    chunks = load_and_split()
    build_vectorstore(chunks)


def ingest_files_batch(uploaded_files):
    valid_files, validation_errors = validate_batch(uploaded_files)

    if not valid_files:
        return {
            "total": len(uploaded_files),
            "saved": 0, "converted": 0,
            "failed": len(uploaded_files),
            "errors": validation_errors,
            "saved_names": [],
        }

    saved_paths, save_errors = save_uploaded_files_batch(valid_files)
    all_errors = validation_errors + save_errors

    converted_paths = convert_batch_to_pdf(saved_paths)
    converted_count = sum(1 for orig, final in zip(saved_paths, converted_paths) if orig != final)

    if converted_paths:
        rebuild_vectorstore()

    return {
        "total":       len(uploaded_files),
        "saved":       len(saved_paths),
        "converted":   converted_count,
        "failed":      len(all_errors),
        "errors":      all_errors,
        "saved_names": [os.path.basename(p) for p in converted_paths],
    }


def format_source_citation(doc):
    meta = doc.metadata
    return {
        "subject": meta.get("subject", "Unknown"),
        "page":    meta.get("page", "?"),
        "snippet": doc.page_content[:250].strip() + "...",
    }


def list_pdf_files(pdf_dir=PDF_DIR):
    if not os.path.exists(pdf_dir):
        return []
    return [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

