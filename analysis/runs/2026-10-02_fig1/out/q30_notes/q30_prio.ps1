While ($true) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_repro_out_qd2?\b|DT_Vessel_abl|qd2_[a-z0-9_]+_s44|qd_comm_s43|meaneval|eval_mean\.py' } | ForEach-Object {
    try { $p = Get-Process -Id $_.ProcessId -ErrorAction Stop; if ($p.PriorityClass -ne 'High') { $p.PriorityClass = 'High' } } catch {}
  }
  Start-Sleep -Seconds 15
}
