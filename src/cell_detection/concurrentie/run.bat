@echo off

echo ************************************************
echo ** Configuratie voor Arno zijn pc ingesteld! **
echo ************************************************

c:
cd "C:\Program Files\CellProfiler"
timeout 2
cellprofiler -c -r -p "D:\Documents\Projects\ProjectBeeldverwerking\src\cell_detection\concurrentie\cellcounter_csv.cppipe" -o "D:\Documents\Projects\ProjectBeeldverwerking\src\cell_detection\concurrentie" --file-list "D:\Documents\Projects\ProjectBeeldverwerking\src\cell_detection\concurrentie\images.txt"
echo CellProfiler done!
