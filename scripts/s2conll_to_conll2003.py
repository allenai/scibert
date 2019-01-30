import plac
import sys

@plac.annotations(
    in_filename=("input file", "positional", None, str),
    out_filename=("output directory", "positional",  None, str))
def main(in_filename, out_filename):
    with open(in_filename) as in_file:
        with open(out_filename, 'w') as out_file:
            for line in in_file:
                line = line.strip()
                if line.startswith('-') or line == "":
                    out_file.write(line)
                else:
                    token_position, surface_form, pos_tag, mention_label, relations, dependency_head_position = line.split('\t')
                    if mention_label in ["U-Entity", "L-Entity"]:
                        mention_label = "I-Entity"
                    out_file.write("\t".join([surface_form, pos_tag, 'O', mention_label]))
                out_file.write("\n")

plac.call(main, sys.argv[1:])
