# C# bindings

Build the native library first, then run:
```bash
dotnet new console -n CudaTaLibExample
# Replace Program.cs with CudaTaLib.cs contents or include as a separate file.
dotnet run --project CudaTaLibExample
```
Ensure the native `cuda_talib_shared` is discoverable (same dir or on PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH).
