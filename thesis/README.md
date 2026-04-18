# Thesis Package

This folder contains an editable VIT-style thesis package for the `Multimodal Disaster Intelligence Platform` project.

## Files

- `metadata.json`: Student names, registration numbers, guide details, school, and submission month.
- `thesis_content.py`: Original chapter text, tables, figures, acronyms, and references used to build the report.
- `build_thesis.py`: Generator that creates the final `.docx` and copies thesis figures into this folder.
- `Multimodal_Disaster_Intelligence_Thesis_VIT.docx`: Generated submission document.

## How to regenerate

From the repository root:

```bash
python3 thesis/build_thesis.py
```

## What you should update before submitting

1. Replace the placeholders in `metadata.json` with your team names, registration numbers, guide name, and dean/program-chair details.
2. Re-run `python3 thesis/build_thesis.py`.
3. Open the generated Word document once and update all fields so the table of contents, list of figures, and list of tables refresh.

## Important note

The report text was written specifically for this repository using the checked-in code, figures, metrics, and documentation. That helps originality, but no one can truthfully guarantee a fixed plagiarism score or AI-detector percentage. Before submission, run the document through the plagiarism or originality workflow approved by your department.
