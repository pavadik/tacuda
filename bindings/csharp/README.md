# C# bindings

The P/Invoke declarations for the tacuda API are generated from
`include/tacuda.h` by `bindings/generate_bindings.py` and committed as
`Tacuda.Generated.cs`.

## Console example

```bash
# Build the native library first
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run the .NET example (from the repository root)
dotnet run --project bindings/csharp/ConsoleExample
```

Ensure the native `tacuda` library is discoverable (e.g. placed next to the
published binary or referenced via `PATH`/`LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`).

## Regenerating

Invoke the generator after editing the C header:

```bash
python bindings/generate_bindings.py
```

CTest verifies that the generated files are up to date.
