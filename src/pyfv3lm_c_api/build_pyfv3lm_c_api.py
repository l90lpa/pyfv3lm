import os
import cffi
import argparse

cmdline = argparse.ArgumentParser(
                    prog='PluginGenerator',
                    description='Generate CFFI Plugin')

cmdline.add_argument('plugin_name')
cmdline.add_argument('build_dir')
cmdline.add_argument('header_file')
cmdline.add_argument('module_file')

args = cmdline.parse_args()
print(args.plugin_name, args.build_dir, args.header_file, args.module_file)

plugin_name = args.plugin_name
build_dir = args.build_dir

with open(args.header_file) as f:
    header = f.read()
header_name = os.path.basename(args.header_file)
# print(header)

with open(args.module_file) as f:
    module = f.read()
# print(module)

generator = cffi.FFI()

generator.embedding_api(header)
generator.set_source(plugin_name, "#include \"{header_name}\"".format(header_name=args.header_file))

generator.embedding_init_code(module)
generator.compile(tmpdir=build_dir, target="lib"+plugin_name+".*", verbose=True)