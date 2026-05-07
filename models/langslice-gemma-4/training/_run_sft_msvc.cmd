@echo off
REM Internal launcher: source MSVC env (so triton/inductor can find cl.exe + headers),
REM then run the SFT training driver. %* is forwarded as CLI args to train_sft.
setlocal
set "PYTHONUTF8=1"
set "HF_HUB_ENABLE_HF_TRANSFER=1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b %errorlevel%
REM Triton ships cuda.lib at <triton>\backends\nvidia\lib\x64\, but its built-in
REM LIBPATH only points at the parent. Append the x64 subdir so cl.exe link finds it.
for /f "delims=" %%P in ('python -c "import triton, os; print(os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'lib', 'x64'))"') do set "TRITON_CUDA_LIB_X64=%%P"
set "LIB=%TRITON_CUDA_LIB_X64%;%LIB%"
python -m sft.train_sft %*
exit /b %errorlevel%
