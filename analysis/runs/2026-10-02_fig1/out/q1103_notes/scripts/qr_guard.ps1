# 2026-11-03 qr4_/qr5_ guard v7 (Task Scheduler 'vessel_qr_guard'). Results are unaffected: priority, launch markers, evictor, log.
#  1) Every ~30 s: our training/eval python (owner OSH) below AboveNormal -> AboveNormal; driver bash below Normal -> Normal.
#     High is not used: 10-06 the server crashed 15 min after High (cause not confirmed, f2_gpuguard.sh note).
#  2) BLOCK markers: per batch GPU, other = used - our estimate. If other > LIM put $O\.gpu<g>_BLOCK so
#     _run_f.sh pick_gpu does not launch there. LIM = 7000 MiB in training stages (author 13:3x: allow the 15.9 GB state
#     the trunk ran in), 11500 in codec/eval/curve stages.
#  3) Evictor (author 13:4x 'take it ourselves'; v7 after the 15:05 overshoot): when a training run of ours is pending
#     (its log does not exist yet), the card has no run of ours, and free < 9600 MiB because of other accounts,
#     start C:\work\qr_hold.py <g> 10240 10 = allocate+touch 10 GB for 10 s, then free and exit. WDDM evicts the
#     other accounts' idle memory to system memory (13:58: 13.4 GB -> 3.2 GB), and our launch takes the room on its
#     next 30 s poll. Cooldown 120 s per card. No steady fill: on WDDM it cannot block others and it paged past
#     physical VRAM (15:05, released after about 1 min).
#  4) Logs state changes to C:\work\qr_guard.log. Ends when both drivers wrote 'driver exit', or after 48 h.
#  This file is ASCII only on purpose (PowerShell 5.1 reads BOM-less scripts as cp949).
$log = 'C:\work\qr_guard.log'
$hold = 'C:\work\qr_hold'
New-Item -ItemType Directory -Force -Path $hold | Out-Null
$PY = 'C:\Users\OSH\anaconda3\envs\mltest\python.exe'
$batches = @(
  @{ name = 'qr4'; seed = 46; dir = 'C:\work\DT_Vessel_v3\Python\_repro_out_qr4'; gpus = @(0, 1); run = 'C:\work\DT_Vessel_v3\run_qr4.log' },
  @{ name = 'qr5'; seed = 47; dir = 'C:\work\DT_Vessel_v3\Python\_repro_out_qr5'; gpus = @(2, 3); run = 'C:\work\DT_Vessel_v3\run_qr5.log' }
)
$CK = 'C:\Users\OSH\VESSEL_checkpoints\comm_intent'; $TOTAL = 16303; $TRAIN_MB = 9300; $OTHER_MB = 1500; $NEED_FREE = 9600
$trainSteps = @('trunk', 'branches', 'ext_branches', 'smoke')
$trainPat = '_ext\.log$|trunk_d6_s\d+\.log$|_(off|comm)_s\d+\.log$'
$pyPat = 'vessel_gym_train|eval_ckpt|eval_scripted|comm_codec|check_branch|check_reward|eval_mean|curve_fig1|summarize_fig1|goals_fig1|preflight|test_golden|_verify_'
$shPat = '_run_f\.sh|qr_run\.sh|qr4_launch|qr5_launch|run_repro\.sh'
$reStart = ' ' + [char]0xC2DC + [char]0xC791 + ' (\S+)$'
function Now { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }
function Say($m) { "$(Now) $m" | Out-File $log -Append -Encoding utf8 }
function Done($f) { (Test-Path $f) -and ((Get-Content $f -Tail 3 -Encoding utf8 -ErrorAction SilentlyContinue) -match 'driver exit') }
$hp = @{}; $lastEvict = @{}
function Alive($g) { $id = $hp[$g]; if (-not $id) { return $false }; return [bool](Get-Process -Id $id -ErrorAction SilentlyContinue) }
function Evict($g, $mb = 10240) {
  $env:CUDA_VISIBLE_DEVICES = "$g"; $env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'; $env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
  $q = Start-Process -FilePath $PY -ArgumentList '-u', 'C:\work\qr_hold.py', "$g", "$mb", '10' -WindowStyle Hidden -PassThru -RedirectStandardError (Join-Path $hold "g$g.err") -RedirectStandardOutput (Join-Path $hold "g$g.out")
  $script:hp[$g] = $q.Id; $script:lastEvict[$g] = Get-Date; Say "evict start GPU $g $mb MB pid $($q.Id)"
}
$prev = @{}; $prevUsers = ''; $t0 = Get-Date; $tick = 0
Say 'guard start (v8: priority + BLOCK + evictor for pending training and eval runs, no steady fill)'
while ($true) {
  $tick++
  if ($tick % 10 -eq 1) {
    $raised = 0
    try {
      foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe' or Name='bash.exe'") {
        $cmd = [string]$p.CommandLine
        $isPy = $p.Name -eq 'python.exe' -and $cmd -match $pyPat
        $isSh = $p.Name -eq 'bash.exe' -and $cmd -match $shPat
        if (-not ($isPy -or $isSh)) { continue }
        $o = Invoke-CimMethod -InputObject $p -MethodName GetOwner -ErrorAction SilentlyContinue
        if ($o.User -ne 'OSH') { continue }
        $q = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
        if (-not $q) { continue }
        if ($isPy -and $q.PriorityClass -in 'Idle','BelowNormal','Normal') { $q.PriorityClass = 'AboveNormal'; $raised++ }
        elseif ($isSh -and $q.PriorityClass -in 'Idle','BelowNormal') { $q.PriorityClass = 'Normal'; $raised++ }
      }
    } catch {}
    if ($raised -gt 0) { Say "priority raised $raised" }
    $users = ((query user 2>$null) | Select-Object -Skip 1 | ForEach-Object { ($_ -replace '^\s*>?','').Split(' ')[0] }) -join ','
    if ($users -ne $prevUsers) { Say "users=$users"; $prevUsers = $users }
  }
  $used = @{}
  foreach ($l in (nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)) { $f = $l -split ','; if ($f.Count -ge 2) { $used[[int]$f[0]] = [int]$f[1] } }
  $allDone = $true
  foreach ($b in $batches) {
    $done = Done $b.run
    if (-not $done) { $allDone = $false }
    $st = ''; $tl = Join-Path $b.dir '_timeline.txt'
    if (Test-Path $tl) { $m = (Get-Content $tl -Tail 40 -Encoding utf8 | Select-String $reStart | Select-Object -Last 1); if ($m) { $st = $m.Matches[0].Groups[1].Value } }
    $isTrain = $st -in $trainSteps
    $LIM = if ($isTrain) { 7000 } else { 11500 }
    $pending = 0
    if ($st -eq 'branches') { foreach ($a in 'off', 'comm') { if (-not (Test-Path (Join-Path $b.dir "$($b.name)_$($a)_s$($b.seed).log"))) { $pending++ } } }
    elseif ($st -eq 'ext_branches') { foreach ($a in 'off', 'comm') { if (-not (Test-Path (Join-Path $b.dir "$($b.name)_$($a)_s$($b.seed)_ext.log"))) { $pending++ } } }
    # v8 (17:2x): pending eval/curve runs (output file not created yet), same rules as _run_f.sh eval_all / curve_all
    $epend = 0
    if ($st -in 'eval', 'ext_eval') { foreach ($k in 'lr', 'lrm') { foreach ($a in 'off', 'comm') { if (-not (Test-Path (Join-Path $b.dir "${k}_$($b.name)_$($a)_s$($b.seed).txt"))) { $epend++ } } } }
    elseif ($st -in 'curve', 'ext_curve') {
      $tot = if ($st -eq 'ext_curve') { '24.0517' } else { '16.0563' }
      foreach ($run in 'trunk_d6', 'off', 'comm') {
        $base = Join-Path $CK "$($b.name)_$($run)_s$($b.seed)"
        $fs = @(Get-ChildItem "$base.step*M.pt" -ErrorAction SilentlyContinue) + @(Get-Item "$base.pt" -ErrorAction SilentlyContinue)
        foreach ($f in $fs) {
          if ($f.Name -match '\.step([0-9.]+)M\.pt$') { $x = $Matches[1] } elseif ($run -eq 'trunk_d6') { $x = '9.04397' } else { $x = $tot }
          if ($run -ne 'trunk_d6' -and $f.Name -notmatch '\.step') { $dup = Get-Item ("$base.step" + [int][Math]::Floor([double]$tot / 2) * 2 + 'M.pt') -ErrorAction SilentlyContinue; if ($dup -and $dup.Length -eq $f.Length) { continue } }
          if (-not (Test-Path (Join-Path $b.dir "curve_$($b.name)_$($run)_s$($b.seed)_${x}M.txt"))) { $epend++ }
        }
      }
    }
    $asked = 0
    foreach ($g in $b.gpus) {
      $mk = @(Get-ChildItem -Path $b.dir -Filter ".gpu${g}_*" -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".gpu${g}_BLOCK" })
      $ours = 0
      foreach ($x in $mk) { if ($x.Name -match $trainPat) { $ours += $TRAIN_MB } else { $ours += $OTHER_MB } }
      $ev = Alive $g
      $other = if ($ev) { 0 } else { [Math]::Max(0, $used[$g] - $ours) }
      $free = $TOTAL - $used[$g]
      $act = ''
      if (-not $done -and $isTrain -and $pending -gt $asked -and $mk.Count -eq 0) {
        $asked++
        $cool = (-not $lastEvict[$g]) -or (((Get-Date) - $lastEvict[$g]).TotalSeconds -gt 120)
        if ($free -lt $NEED_FREE -and -not $ev -and $cool) { Evict $g; $act = 'evict' }
      }
      if (-not $done -and -not $isTrain -and $epend -gt $asked -and $mk.Count -eq 0 -and $free -lt 4000 -and -not $ev) {   # v8b: only on a card with no run of ours; allocate 7000 MB total (> free) so WDDM pushes other accounts out
        $asked++
        $cool = (-not $lastEvict[$g]) -or (((Get-Date) - $lastEvict[$g]).TotalSeconds -gt 120)
        if ($cool) { Evict $g 7000; $act = 'evict-eval' }
      }
      $blk = Join-Path $b.dir ".gpu${g}_BLOCK"
      $has = Test-Path $blk
      if (-not $ev) {
        if (-not $done -and -not $has -and $other -gt $LIM) { New-Item -ItemType File -Path $blk -Force | Out-Null; $has = $true }
        elseif ($has -and ($done -or $other -le $LIM)) { Remove-Item $blk -Force -ErrorAction SilentlyContinue; $has = $false }
      }
      $state = if ($has) { 'BLOCK' } else { 'ok' }
      $key = "$state/$([int]($other / 1000))/$LIM/$pending/$epend/$ev/$act"
      if ($prev["$($b.name)g$g"] -ne $key) { Say "$($b.name) GPU $g step=$st pend=$pending epend=$epend lim=$LIM used=$($used[$g]) ours=$ours other=$other evictor=$ev $act -> $state (markers: $(($mk | ForEach-Object Name) -join ' '))"; $prev["$($b.name)g$g"] = $key }
    }
  }
  if ($allDone) { Say 'both drivers exited - guard end'; break }
  if (((Get-Date) - $t0).TotalHours -gt 48) { Say '48 h - guard end'; break }
  Start-Sleep -Seconds 3
}
