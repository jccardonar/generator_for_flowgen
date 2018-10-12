'''
Setups a reproducible network scenario and generates scripts that can be run with flowgen to simulate it.
TODO: We really need a netflow v9 generator, for ipv6 and direction of flows
'''
import argparse
import csv
import ipaddress
import itertools
import logging
import os
import shutil
import stat
import sys

import numpy as np

# LOGGING
# =====
logger = logging.getLogger(__name__)
logger.level = logging.INFO
handlers = set()

# file handler
#file_handler = logging.FileHandler(constants.FILE_LOG)
#file_handler.setLevel(logging.WARNING)
#handlers.add(file_handler)

# stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
handlers.add(stream_handler)

if not logger.handlers:
    for handler in handlers:
        logger.addHandler(handler)


# FUNCTIONS
# ========
# CHECK FUNCTIONS
# -------------
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid zero or positive int value" % value)
    return ivalue


def check_asn(value):
    ivalue = int(value)
    if ivalue <= 0 or ivalue >= 2**32:
        raise argparse.ArgumentTypeError("%s is an invalid ASN" % value)
    return ivalue


max_number_int = 20


def check_interfaces(value):
    ivalue = int(value)
    if ivalue <= 1 or ivalue > max_number_int:
        raise argparse.ArgumentTypeError(
            "%s is an invalid number of interfaces. Interfaces should be between 2 and %s."
            % (value, max_number_int))
    return ivalue


# OTHER FUNCTIONS
# ----------------
# function to generate asn, let's keep the random ASNs 2-byte
def asn_gen():
    return np.random.randint(0, 2**16)


# ELEMENTS DISTRIBUTION FUNCTIONS
# -------------------


def distribute_ases_among_interfaces(neigh_to_interface, interfaces_to_neigh,
                                     prefixes_per_asns, asn_per_prefix):
    '''
    Distributes external ASes to external interfaces.
    Note that input variables are dicts which are filled by reference.
    '''
    first_interfaces = np.arange(1, arguments.number_interfaces + 1)
    np.random.shuffle(first_interfaces)
    first_interfaces = first_interfaces[0:arguments.number_external_asn]
    list_neightors = list(neighboring_asns)

    for counter, this_interface in enumerate(first_interfaces):
        this_neighbor = list_neightors[counter]
        neigh_to_interface[this_neighbor] = this_interface
        interfaces_to_neigh.setdefault(this_interface,
                                       set()).add(this_neighbor)

    # distribute the remainding neighboring ASes
    for this_neighbor in neighboring_asns - set(neigh_to_interface):
        this_interface = np.random.randint(1, arguments.number_interfaces + 1)
        neigh_to_interface[this_neighbor] = this_interface
        interfaces_to_neigh.setdefault(this_interface,
                                       set()).add(this_neighbor)

    # make sure that no neighboring_as is in interface 0, which is the "local" flow
    if 0 in interfaces_to_neigh:
        raise Exception(
            "Wrong interface assignment. Neighboring AS {} in interface 0".
            format(interfaces_to_neigh[0]))

    interfaces_to_neigh[0] = {arguments.local_asn}
    neigh_to_interface[arguments.local_asn] = 0


def divide_numbers_in_ranges(num, partition_size):
    '''
    Distributes a number of events into partitions.
    '''
    # generate n random numbers, find the cumsum (so it is monotinically increasing), and scale them to sum the total amount of flows
    partition_list_start = np.random.uniform(size=partition_size)
    partition_list_withzero = np.insert(partition_list_start, 0, 0)
    partition_list_cumsum = np.cumsum(partition_list_withzero)
    partition_list_normalized = partition_list_cumsum * num / partition_list_cumsum[
        -1]

    # convert it to int, and find the diff, then assign number of flows to interfaces (making sure that the sum is equal to the total)
    # the top number (partition_list[-1]) should be arguments.number_of_flows, but rounding errors might occur, so make sure that it is the number
    if abs(partition_list_normalized[-1] - num) > 0.001:
        raise Exception(
            "Problem in flow to interface pair assignment. Total flows not equal to required number"
        )

    partition_list_normalized[-1] = num

    partition_list_norm_int = partition_list_normalized.astype(np.int)

    partition_list = np.diff(partition_list_norm_int)
    if len(partition_list) != partition_size:
        raise Exception(
            "Error calculating number of flows per interface pair. Len does not match"
        )
    if sum(partition_list) != num:
        raise Exception(
            "Number of assigned flows per interface pair does not sum the required number: assigned {}, required {}"
            .format(sum(partition_list), num))
    if not np.issubdtype(partition_list.dtype, np.integer):
        raise Exception("Returned array does not contain integers. Mistake")

    return partition_list


def test_partition_code(generation_function, number_to_divide, partition_size):
    '''
     after playing with this a bit, you realize that the first value tends to have
     less flows, and the last one more. Good for now though.
    '''
    sum_partition = np.zeros(partition_size)
    for n in range(0, 1000):
        new_partition = generation_function(number_to_divide, partition_size)
        sum_partition = sum_partition + np.array(new_partition)
    return sum_partition / 1000


# SET UP PARSER
# ===========
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c_ip",
    "--collector_ip",
    help="IP address of the Netflow collector",
    required=True)
parser.add_argument(
    "-c_port",
    "--collector_port",
    type=check_positive,
    help="Port in which the Netflow collector listens.",
    required=True)
parser.add_argument(
    "-nflows",
    "--number_of_flows",
    type=check_positive,
    help=
    "Number of flows to generate. Flows will be counted based on src and dst prefix, not over port, src/dst host, etc.",
    required=True)
parser.add_argument(
    "-interval",
    type=check_positive,
    help="Interval for flow generation in seconds. Default: 60.",
    default=60)
parser.add_argument(
    "-samples",
    type=check_positive_or_zero,
    help="Number of intervals to execute. 0 for unlimited. Default: 1.",
    default=1)
parser.add_argument("-log", "--log_file", help="Log file.")
parser.add_argument(
    "-lasn",
    "--local_asn",
    type=check_positive,
    help="Defines the local ASN. If not given, it is chosen at random")
parser.add_argument(
    "-n_extasn",
    "--number_external_asn",
    type=check_asn,
    help="Number of external ASNs. Default 10",
    default=10)
parser.add_argument(
    "-n_interfaces",
    "--number_interfaces",
    type=check_positive,
    help="Number of interfaces, between 2 and {}. Default 5".format(
        max_number_int),
    default=5)
parser.add_argument(
    "-seed",
    type=check_positive_or_zero,
    help="Seed for the random number generator.")
arguments = parser.parse_args()

# START SCENARIO
# ========
# DEFINE MISSING VARIABLES
# ------------
# define seed
if arguments.seed is None:
    arguments.seed = np.random.randint(2**32 - 1)
    logger.debug("Selected seed: {}".format(arguments.seed))

np.random.seed(arguments.seed)

# select local as
if arguments.local_asn is None:
    arguments.local_asn = asn_gen()
    logger.debug("Local ASN: {}".format(arguments.local_asn))

collector_ip = arguments.collector_ip
collector_port = arguments.collector_port

logger.info("Parameters %s", arguments)

# SELECT NEIGHBORING ASNs, THEIR POSITION
# -------------------------------
neighboring_asns = set(
    (asn_gen() for n in range(0, arguments.number_external_asn)))

# assign neighboring ases to external interfaces.
# We first make sure that all interfaces have one AS.

# if there are ASN left, we distribute them randomly
# Note: there is a better way using the random library, but we must keep the seed.
neigh_to_interface = {}
interfaces_to_neigh = {}

prefixes_per_asns = {}
asn_per_prefix = {}

distribute_ases_among_interfaces(neigh_to_interface, interfaces_to_neigh,
                                 prefixes_per_asns, asn_per_prefix)

# the scenario is setup here
logger.info(neigh_to_interface)

# FLOW GENERATION
# =============

# DISTRIBUTE FLOWS PER INTERFACES
# -------------------
# find interface pairs
interface_pairs = list(itertools.permutations(interfaces_to_neigh, 2))

# divide the number of flows across the interface pairs
# we could go right at dividing it over ASN pairs, but this makes sure the flows are more or less distributed over interface pairs
flows_per_pair_number = divide_numbers_in_ranges(arguments.number_of_flows,
                                                 len(interface_pairs))
flows_per_pair = dict(zip(interface_pairs, flows_per_pair_number))

# DISTRIBUTE FLOWS ACROSS ASN PAIRS
# -------------------------
asn_pairs_flows = {}
for interface_1, interface_2 in interface_pairs:
    if interface_1 == interface_2:
        raise Exception("interface pairs should not be equal")
    # find pairs of asns
    asn_pairs_for_this_int = list(
        itertools.product(interfaces_to_neigh[interface_1],
                          interfaces_to_neigh[interface_2]))
    # divide the number of flows per interface pairs over that
    flows_per_these_asns_pairs = divide_numbers_in_ranges(
        flows_per_pair[(interface_1, interface_2)],
        len(asn_pairs_for_this_int))
    asn_pairs_flows.update(
        dict(zip(asn_pairs_for_this_int, flows_per_these_asns_pairs)))

if sum(asn_pairs_flows.values()) != arguments.number_of_flows:
    raise Exception(
        "Number of flows do not correspond to asn pairs. Expected {}, got {}.".
        format(arguments.number_of_flows, sum(asn_pairs_flows.values())))

# CREATE TRAFFIC DISTRIBUTION
# -----------------------
# let us build a mean traffic distribution that resembles more or less a log-normal (heavy tail)
mean_packets_dist = np.random.lognormal(
    mean=1, sigma=1, size=arguments.number_of_flows)
mean_octet_dist = np.random.lognormal(
    mean=1, sigma=1, size=arguments.number_of_flows)

# DISTRIBUTE FLOWS ACROSS PREFIXES
# -----------------------
# prefixes per asns are assigned at the /16 level.
# A pair of prefixes per AS can sustain 256*256 flows (at the /24 level).

const_numbers_per_octet = 254


def ip_sixteen_prefix_generator():
    first_octet = list(range(1, const_numbers_per_octet + 1))
    second_octet = list(range(1, const_numbers_per_octet + 1))
    if len(first_octet) != const_numbers_per_octet or len(
            second_octet) != const_numbers_per_octet:
        raise Exception(
            "Number of octets in generated prefix bad created, check generator"
        )
    octet_combinations = list(itertools.product(first_octet, second_octet))
    np.random.shuffle(octet_combinations)
    for octet_combination in octet_combinations:
        yield ipaddress.ip_network('{}.{}.0.0/16'.format(
            octet_combination[0], octet_combination[1]))


ip_sixteen_generator = ip_sixteen_prefix_generator()

prefix_flows = {}
rest_of_flows = {}

# Create folder  where ASN flows are stored. We create a file per ASN pair.
folder_tmp = "asn_flows_descriptors"
if os.path.isdir(folder_tmp):
    shutil.rmtree(folder_tmp)
os.makedirs(folder_tmp)

cache_prefix_str = {}
flow_n = -1
commands = []
for (asn_1, asn_2), this_num_of_flows in asn_pairs_flows.items():

    origin_interface = neigh_to_interface[asn_1]
    destination_interface = neigh_to_interface[asn_2]
    origin_asn = asn_1
    destination_asn = asn_2

    this_num_of_flows = int(this_num_of_flows)
    if this_num_of_flows <= 0:
        continue
    # let us add prefixes to each AS until we can get enough number to fulfill the number of flows
    # let us do it iteretively first, TODO: do it with a faster process
    while len(prefixes_per_asns.get(asn_1, set())) * 256 * len(
            prefixes_per_asns.get(asn_2, set())) * 256 < this_num_of_flows:
        asn_with_less_prefixes = asn_1 if len(
            prefixes_per_asns.get(asn_1, set())) < len(
                prefixes_per_asns.get(asn_2, set())) else asn_2
        try:
            new_prefix = next(ip_sixteen_generator)
        except StopIteration:
            raise Exception("The software could not fill the number of flows.")
        except:
            raise Exception("Problem at generating /16 prefixes")

        prefixes_per_asns.setdefault(asn_with_less_prefixes,
                                     set()).add(new_prefix)
        if new_prefix in asn_per_prefix:
            raise Exception("Repetated generated prefix")
        asn_per_prefix[new_prefix] = asn_with_less_prefixes

    # To generate the prefixes, we need a couple of auxiliary generators

    generator_of_subnets_generators = lambda x: (prefix.subnets(8) for prefix in x)
    product_iterator = itertools.product(
        itertools.chain.from_iterable(
            generator_of_subnets_generators(prefixes_per_asns[asn_1])),
        itertools.chain.from_iterable(
            generator_of_subnets_generators(prefixes_per_asns[asn_2])))
    rest_of_flows[(origin_asn, destination_asn)] = {(origin_interface,
                                                     destination_interface)}
    for prefix_1, prefix_2 in itertools.islice(product_iterator,
                                               this_num_of_flows):
        flow_n += 1
        packets = int(mean_packets_dist[flow_n])
        octets = int(mean_octet_dist[flow_n])
        if prefix_1 in cache_prefix_str:
            origin_prefix_divided = cache_prefix_str[prefix_1]
        else:
            origin_prefix_divided = tuple(str(prefix_1).split('.')[0:3])
        if prefix_2 in cache_prefix_str:
            destination_prefix_divided = cache_prefix_str[prefix_2]
        else:
            destination_prefix_divided = tuple(str(prefix_2).split('.')[0:3])

        prefix_flows.setdefault((origin_asn, destination_asn), set()).update(
            ((origin_prefix_divided + destination_prefix_divided +
              (packets, octets)), ))

    file_name = folder_tmp + "/" + "{}_{}_{}_{}".format(
        origin_asn, destination_asn, origin_interface, destination_interface)
    # let us write the file.
    with open(file_name, 'w') as file_h:
        csvwriter = csv.writer(file_h)
        for line in zip(*prefix_flows[(origin_asn, destination_asn)]):
            csvwriter.writerow(line)
    commands.append(
        ("flowgen", collector_ip, "-p", str(collector_port), "-i", "1",
         "--dstaddr",
         ".".join(["%{}[{}]".format(file_name, x)
                   for x in [3, 4, 5]] + ["1:254"]), "-n",
         str(len(prefix_flows[(origin_asn, destination_asn)])), "--srcas",
         str(origin_asn), "--dstas", str(destination_asn), "--srcaddr",
         ".".join(["%{}[{}]".format(file_name, x)
                   for x in [0, 1, 2]] + ["1:254"]), "--inputif",
         str(origin_interface), "--outputif",
         str(destination_interface), '--packets', "%{}[{}]".format(
             file_name, 6), '--octets', "%{}[{}]".format(file_name, 7), '--m'))

# WRITE FLOWGET BASH SCRIPT
# =============
with open("commands", 'w') as file_h:
    file_h.write("#!/usr/bin/env bash\n")
    file_h.write('trap "set +x; sleep 0.07; set -x" DEBUG\n')
    for command in commands:
        file_h.write(' '.join(command) + '\n')

st = os.stat('commands')
os.chmod('commands', st.st_mode | stat.S_IEXEC)
