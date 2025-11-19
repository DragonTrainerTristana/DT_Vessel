import sys
import importlib.metadata

print("=" * 60)
print("ML-Agents Version Compatibility Check")
print("=" * 60)

# Check Python version
print(f"\n[Python Version] {sys.version}")

# Check mlagents version
try:
    mlagents_version = importlib.metadata.version('mlagents')
    print(f"[mlagents Version] {mlagents_version}")
except:
    print("[mlagents Version] Not installed")

print("\n" + "=" * 60)
print("Compatibility Matrix:")
print("=" * 60)
print("\nUnity ML-Agents 4.0.0 requires:")
print("  - Python 3.10.x")
print("  - mlagents 1.1.0")

print("\nUnity ML-Agents 2.3.0 requires:")
print("  - Python 3.8.x - 3.10.x")
print("  - mlagents 0.28.0")

print("\nYour current setup:")
python_major = sys.version_info.major
python_minor = sys.version_info.minor

if python_major == 3 and python_minor == 12:
    print(f"  - Python {python_major}.{python_minor} -> Compatible with mlagents 0.28.0 only")
    print("  - Unity ML-Agents 4.0.0 -> INCOMPATIBLE")
    print("\n[RECOMMENDATION]")
    print("Option 1: Switch to Python 3.10 + mlagents 1.1.0")
    print("         Command: py -3.10 test_unity_connection.py")
    print("\nOption 2: Downgrade Unity ML-Agents to 2.3.0")
    print("         (In Unity Package Manager)")
elif python_major == 3 and python_minor == 10:
    print(f"  - Python {python_major}.{python_minor} -> Compatible with mlagents 1.1.0")
    print("  - Unity ML-Agents 4.0.0 -> COMPATIBLE")
    print("\n[STATUS] Version compatibility OK!")
    print("However, Unity gRPC DLL issue must be fixed first.")

print("\n" + "=" * 60)
print("Main Issue: Unity gRPC DLL Loading Error")
print("=" * 60)
print("\nThe core problem is not Python version, but Unity's missing")
print("grpc_csharp_ext.x64.dll file. This prevents ANY Python version")
print("from connecting to Unity.")
print("\n[SOLUTION] Fix Unity ML-Agents package:")
print("1. In Unity: Window > Package Manager")
print("2. Remove ML-Agents package")
print("3. Re-add ML-Agents 4.0.0")
print("4. Restart Unity Editor")