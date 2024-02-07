import glob
import subprocess
import tempfile
import os
import pathlib
import argparse

def is_gfortran_installed():
    try:
        subprocess.run(['gfortran', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def gfortran_preprocessor(file, *, definitions: [str] = None, includes: [str] = None):
    preprocessed_file = pathlib.Path('preprocessed_' + file.name)

    program = ['gfortran']
    args = ['-P', '-E', '-cpp', file, '-o', preprocessed_file]
    
    if definitions:
        args.extend(['-D'+definition for definition in definitions])

    if includes:
        args.extend(['-I'+include for include in includes])

    subprocess.run([*program, *args])

    return preprocessed_file

def f90wrap(files, module_name, ext_module_name, *, kind_map=None, only=None, skip=None):
    assert (only == None) or (skip == None)
    
    program = ['python', '-m', 'f90wrap', '--f90wrap']
    args = [*files, '--mod-name', module_name, '--f90-mod-name', ext_module_name, '--default-to-inout']
    
    if kind_map:
        args.append('--kind-map')
        args.append(str(kind_map))

    if only:
        args.append('--only')
        args.extend(only)
    elif skip:
        args.append('--skip')
        args.extend(skip)

    subprocess.run([*program, *args])
    
    fortran_wrapped_files = [file.with_name('f90wrap_' + file.stem + ".f90") for file in files]
    python_wrapped_file = pathlib.Path(module_name + '.py')
    
    return fortran_wrapped_files, python_wrapped_file


def f2py_f90wrap(files, ext_module_name, *, kind_map=None):

    program = ['python', '-m', 'f90wrap', '--f2py-f90wrap']
    args = [*files, '-m', ext_module_name, '--lower', '--quiet']

    if kind_map:
        args.append('--f2cmap')
        args.append(str(kind_map))

    subprocess.run([*program, *args])

    cpython_wrapper_file = pathlib.Path(ext_module_name + 'module.c')

    return cpython_wrapper_file
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=pathlib.Path, nargs='+', help='files to process')
    parser.add_argument('-p', '--package-name', type=str, help='package name')
    parser.add_argument('-k', '--kind-map', type=pathlib.Path, help='Fortran to C kind map file (e.g. `.f2py_f2cmap`)')
    parser.add_argument('-D', type=str, action='append', help='preprocessor defintion (can be supplied multiple times)')
    parser.add_argument('-I', type=str, action='append', help='include path (can be supplied multiple times)')
    subroutine_filters = parser.add_argument_group('subroutine filters')
    subroutine_filters = subroutine_filters.add_mutually_exclusive_group()
    subroutine_filters.add_argument('--only', type=str, nargs='+', help='space separated list of the only subroutine names to wrap')
    subroutine_filters.add_argument('--skip', type=str, nargs='+', help='space separated list all the subroutine names to skip wrapping')

    args = parser.parse_args()

    ext_module_name = '_'+args.package_name
    
    print(args.files)
    print(args.package_name)
    print(args.kind_map)
    print(ext_module_name)
    print(args.D)
    print(args.I)
    print(args.only)
    print(args.skip)

    if is_gfortran_installed():
        preprocessor = gfortran_preprocessor
    else:
        print('ERROR: a Fortran preprocessor could not be detected')
        exit(-1)

    preprocessed_files = [preprocessor(file, definitions=args.D, includes=args.I) for file in args.files]
    
    f90wrap_fortran_files, f90wrap_python_file = f90wrap(preprocessed_files, args.package_name, ext_module_name,
                                                         kind_map=args.kind_map, only=args.only, skip=args.skip)

    f2py_c_file = f2py_f90wrap(f90wrap_fortran_files, ext_module_name, kind_map=args.kind_map)

    # print(f"f2py_c_file={f2py_c_file}")
    # print(f"f90wrap_python_file={f90wrap_python_file}")