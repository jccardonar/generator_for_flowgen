This code constructs a bash script that can simulate the "netflow v5" flows of a randomized, and arbitrarly large, network.

It requires [flowgen](https://github.com/mshindo/NetFlow-Generator/), specifically its [modified version ](https://github.com/jccardonar/NetFlow-Generator/), to generate the flows.

run `python netflow_wrapper.py --help` for instructions. 

The script generates a folder called "asn_flows_descriptors/" and a executable bash script "commands.sh" that can be run to generate the flows.

Use [parallel](https://www.gnu.org/software/parallel/) to run the flow commands in parallel. As an alternative, modify the script to add a "&" at the end of each command to set it on the background.

Only tested in python 2.7 and 3.6
