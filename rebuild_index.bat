@echo off
echo ========================================
echo Rebuilding RAG Index with New Metadata
echo ========================================
echo.

echo Step 1: Deleting old .pkl files...
del /Q db\*.pkl 2>nul
if %errorlevel% equ 0 (
    echo   [OK] Old index files deleted
) else (
    echo   [INFO] No old index files found
)
echo.

echo Step 2: Running pipeline to rebuild index...
python pipeline.py
echo.

echo ========================================
echo Rebuild Complete!
echo ========================================
echo.
echo Check db\ folder for new .pkl files with metadata
echo Check db\results.csv for updated predictions
echo.
pause
