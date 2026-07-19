# build.ps1 [-Main <src.c>] [-ExeName <name.exe>]
# Compiles a main .c together with all libllsm2/ciglet/pyin/gvps sources.
# Pure ASCII: derives all paths from $PSScriptRoot so the file needs no BOM
# and survives editors that save UTF-8 without BOM (PS 5.1 reads .ps1 as ANSI).
# Uses a UTF-16LE+BOM cl response file (in ASCII TEMP) so cl reads the
# non-ASCII source paths correctly. FP_TYPE=float matches llsm_runner.vcxproj.
param(
  [string]$Main = "llsm_utau.c",
  [string]$ExeName = ""
)
if ($ExeName -eq "") { $ExeName = "WCSNDM.exe" }   # engine product name

$HERE   = $PSScriptRoot
$root   = Split-Path $HERE -Parent                                  # ...\<project root>
$LLSM   = Join-Path $root "LLSM\llsm 1009\llsm2\llsm2\libllsm2-master"
$RUNNER = Join-Path $root "LLSM\llsm 1009\llsm2\llsm_runner\llsm_runner"
$OUT    = Join-Path $HERE "build"
if (-not (Test-Path $OUT)) { New-Item -ItemType Directory -Path $OUT | Out-Null }

$rsp = Join-Path $env:TEMP "llsm_utau_cl.rsp"
# /MT = static CRT -> self-contained exe, callable by UTAU/OpenUTAU subprocess (no VS DLLs).
$lines = @("/nologo","/MT","/O2","/DFP_TYPE=float","/D_CRT_SECURE_NO_WARNINGS",
 "/I`"$LLSM`"","/I`"$LLSM\external`"","/I`"$LLSM\external\ciglet`"","/I`"$LLSM\external\ciglet\external`"",
 "/I`"$LLSM\external\libgvps`"","/I`"$LLSM\external\libpyin`"","/I`"$RUNNER`"","`"$HERE\$Main`"")
foreach ($s in @("coder.c","container.c","dsputils.c","frame.c","layer0.c","layer1.c","llsmutils.c",
 "external\ciglet\ciglet.c","external\ciglet\external\fast_median.c","external\ciglet\external\fftsg_h.c",
 "external\ciglet\external\wavfile.c","external\libgvps\gvps_full.c","external\libgvps\gvps_obsrv.c",
 "external\libgvps\gvps_sampled.c","external\libgvps\gvps_variable.c","external\libpyin\math-funcs.c",
 "external\libpyin\pyin.c","external\libpyin\yin.c")) { $lines += "`"$LLSM\$s`"" }
# LLSM1 backend: compiled separately (own dir first + /FI prefix header to namespace
# symbols), then its .obj are linked into the main exe. ASCII-only (PS 5.1 reads .ps1 as ANSI).
$L1SRC = Join-Path $HERE "llsm1_src"
$L1OUT = Join-Path $OUT "l1"
if (-not (Test-Path $L1OUT)) { New-Item -ItemType Directory -Path $L1OUT | Out-Null }
$L1FILES = @("math-funcs.c","llsm-layer0.c","llsm-layer1.c","envelope.c","l1_path.c")
foreach ($f in $L1FILES) { $lines += "`"$L1OUT\$([System.IO.Path]::GetFileNameWithoutExtension($f)).obj`"" }

$lines += "/Fo`"$OUT\\`""
$lines += "/Fe`"$OUT\$ExeName`""

$enc = New-Object System.Text.UnicodeEncoding($false, $true)
[System.IO.File]::WriteAllText($rsp, ([string]::Join("`r`n", $lines) + "`r`n"), $enc)
Write-Host "[build] rsp: $rsp"

# LLSM1 precompile rsp (UTF-16LE+BOM, holds non-ASCII paths)
$rsp1 = Join-Path $env:TEMP "llsm1_cl.rsp"
$l1lines = @("/nologo","/c","/TC","/O2","/MT","/DFP_TYPE=float","/D_CRT_SECURE_NO_WARNINGS","/D_USE_MATH_DEFINES",
 "/FI`"$L1SRC\l1_prefix.h`"","/I`"$L1SRC`"","/I`"$HERE`"","/I`"$LLSM`"","/I`"$LLSM\external`"",
 "/I`"$LLSM\external\ciglet`"","/I`"$LLSM\external\ciglet\external`"","/I`"$LLSM\external\libpyin`"","/I`"$LLSM\external\libgvps`"")
foreach ($f in $L1FILES) { $l1lines += "`"$L1SRC\$f`"" }
$l1lines += "/Fo`"$L1OUT\\`""
[System.IO.File]::WriteAllText($rsp1, ([string]::Join("`r`n", $l1lines) + "`r`n"), $enc)

$envText = & cmd /c "`"F:\VS\VC\Auxiliary\Build\vcvarsall.bat`" x64 >nul 2>&1 & set"
foreach ($l in $envText) { if ($l -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:"+$matches[1]) -Value $matches[2] } }
try { [Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding(936) } catch {}

Write-Host "[build] compiling LLSM1 backend ..."
& cl.exe "@$rsp1" *> "$L1OUT\compile.log"
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] LLSM1 FAILED:"
  Get-Content "$L1OUT\compile.log" -Encoding Default | Select-String -Pattern ": error|: fatal" | Select-Object -First 20 | ForEach-Object { $_.Line }
}

$log = "$OUT\compile.log"
Write-Host "[build] compiling ..."
& cl.exe "@$rsp" *> $log
$code = $LASTEXITCODE

if ((Test-Path "$OUT\$ExeName") -and $code -eq 0) {
  Write-Host ("[build] OK -> " + $ExeName + " (" + (Get-Item "$OUT\$ExeName").Length + " bytes)")
} else {
  Write-Host "[build] FAILED (exit $code). Errors:"
  Get-Content $log -Encoding Default | Select-String -Pattern ": error|: fatal" | Select-Object -First 30 | ForEach-Object { $_.Line }
}
