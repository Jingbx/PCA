# # Read the process ID from the file
# $processId = Get-Content -Path "G:\DJL\DALF_v1\zTrainShell\train_pid.txt"

# # Stop the process using the process ID
# Stop-Process -Id $processId -Force

# # Output a message indicating the process has been stopped
# Write-Output "Stopped train.py with PID $processId"

# Read the process ID from the file
$processId = Get-Content -Path "zTrainShell\train_pid.txt"

# Stop the process using the process ID
Stop-Process -Id $processId -Force

# Output a message indicating the process has been stopped
Write-Output "Stopped train.py with PID $processId"
