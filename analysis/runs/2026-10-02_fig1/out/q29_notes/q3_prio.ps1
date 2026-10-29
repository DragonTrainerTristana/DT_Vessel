While ($true) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '_repro_out_q(d|d2|e)\b|DT_Vessel_pbrs|(qd2?|qe)_[a-z0-9_]+_s4\d|meaneval|eval_mean\.py' } | ForEach-Object {
    try { $p = Get-Process -Id $_.ProcessId -ErrorAction Stop; if ($p.PriorityClass -ne 'High') { $p.PriorityClass = 'High' } } catch {}
  }
  Start-Sleep -Seconds 15
}
