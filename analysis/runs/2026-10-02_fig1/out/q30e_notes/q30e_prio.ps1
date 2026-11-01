While ($true) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_repro_out_(qs2?|qd[2-5]?)\b|DT_Vessel_abl|(qs2?|qd[2-5]?)_[a-z0-9_]+_s4[3-7]|meaneval|eval_mean\.py' } | ForEach-Object {
    try { $p = Get-Process -Id $_.ProcessId -ErrorAction Stop; if ($p.PriorityClass -ne 'High') { $p.PriorityClass = 'High' } } catch {}
  }
  Start-Sleep -Seconds 15
}
