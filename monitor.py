import subprocess
import numpy as np
import tensorflow as tf
import time

deltadic = {1 : {'rx pkts': 0, 'rx bytes': 0, 'tx pkts': 0, 'tx bytes': 0},
          2 : {'rx pkts': 0, 'rx bytes': 0, 'tx pkts': 0, 'tx bytes': 0},
          3 : {'rx pkts': 0, 'rx bytes': 0, 'tx pkts': 0, 'tx bytes': 0}}

def parse_ovs_dump_ports_output(output):
    """Parses the output of 'ovs-ofctl dump-ports s1' command and extracts important values."""
    lines = output.splitlines()[1:]
    lines = [' '.join(lines[i:i+2]) for i in range(0,len(lines)-1,2)]

    values = {}
    for line in lines:
        # print(line,'####')
        
        if "port " in line and 'LOCAL' not in line:
            port_name = line.split("port ")[1].split(":")[0].strip()
            port_name = int(port_name[-2])
            values[port_name] = {}
            tmp = line.split('            ')
            # print(tmp)

            rx_packets = int(tmp[0].split("rx pkts=")[1].split(",")[0].strip())
            values[port_name]["rx pkts"] = rx_packets

            rx_bytes = int(tmp[0].split("bytes=")[1].split(",")[0].strip())
            values[port_name]["rx bytes"] = rx_bytes

            tx_packets = int(tmp[1].split("tx pkts=")[1].split(",")[0].strip())
            values[port_name]["tx pkts"] = rx_packets

            tx_bytes = int(tmp[1].split("bytes=")[1].split(",")[0].strip())
            values[port_name]["tx bytes"] = rx_bytes
    return values

def get_dump_tables(inp):

    tmp = inp.splitlines()[2]

    active = int(tmp.split("active=")[1].split(",")[0].strip())
    lookup = int(tmp.split("lookup=")[1].split(",")[0].strip())
    matched = int(tmp.split("matched=")[1].split(",")[0].strip())

    
    return [active,lookup,matched]


model = tf.keras.models.load_model("netmodel")
while (True):
    time.sleep(5.0)
    portdata = subprocess.getoutput("sudo ovs-ofct dump-ports s1")
    tabledata = subprocess.getoutput("sudo ovs-ofct dump-tables s1")

    pdata = parse_ovs_dump_ports_output(portdata)
    tdata = get_dump_tables(tabledata)

    pdatalist = []
    for port, data in pdata.items():
        tmp = []
        for f, d in data.items():
            tmp.append(pdata[port][f])
        pdatalist.append(tmp)

    deltas = []
    for port, data in pdata.items():
        tmp = []
        for f, d in data.items():
            tmp.append(pdata[port][f] - deltadic[port][f])
        deltas.append(tmp)

    deltadic = pdata

    data = np.array([pdatalist[i]+deltas[i]+[list(pdata.keys())[i]]+tdata for i in range(len(deltas))])
    data = (data-data.min())/(data.max()-data.min())
    preds = np.argmax(model.predict(data), axis=1)
    print(preds)

