# # Change to the appropriate directory
# cd G:\DJL\DALF_v1

# # Start the Python script and get the process object
# $process = Start-Process -WindowStyle Hidden -FilePath "python" -ArgumentList "train.py > zTrainShell\train.log 2>&1" -PassThru

# # Get the process ID
# $processId = $process.Id

# # Output the process ID to a file
# $processId | Out-File -FilePath "zTrainShell\train_pid.txt"

# # Output the process ID to the console for reference
# Write-Output "Started train.py with PID $processId"

# Change to the appropriate directory
# cd G:\DJL\GLADv1

# Log the current directory
Write-Output "Current directory: $(Get-Location)"

# Start the Python script and get the process object
$process = Start-Process -WindowStyle Hidden -FilePath "python" -ArgumentList "train_network.py" -RedirectStandardOutput "zTrainShell\train_total_ablation_no_lsrm.log" -RedirectStandardError "zTrainShell\train_process_ts-fl_ablation_no_lsrm.log" -PassThru

# Get the process ID
$processId = $process.Id

# Output the process ID to a file
$processId | Out-File -FilePath "zTrainShell\train_pid.txt"

# Output the process ID to the console for reference
Write-Output "Started train.py with PID $processId"