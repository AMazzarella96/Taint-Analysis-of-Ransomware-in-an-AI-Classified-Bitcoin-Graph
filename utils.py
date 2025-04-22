import pandas as pd
import numpy as np
import gc
import requests
import sklearn
from datetime import datetime
from collections import defaultdict
from sklearn.utils import resample
import statistics
import time
import socks
import socket
from stem import Signal
from stem.control import Controller
from itertools import cycle
from requests.exceptions import HTTPError


controller = Controller.from_port(port=9052)

def connectTor():
    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", 9150, True)
    socket.socket = socks.socksocket

def renewTor():
    controller.authenticate("TesiMazzarella123")
    controller.signal(Signal.NEWNYM)

def to_btc(n):
    formatted_n = n / 100000000
    return formatted_n

def get_statistics(address):
    url = f"https://blockchain.info/rawaddr/{address}"  
    restart = False
    while True:
        if restart:
            renewTor()
            connectTor()
            restart = False
        try:
            time.sleep(10)
            response = requests.get(url)
            response.raise_for_status()  
            data = response.json()
            break
        except HTTPError as http_err:
            if response.status_code == 429:  # Too Many Requests
                print(f"{http_err}")
                restart = True
                #time.sleep(10)
                #renewTor()
                #connectTor()
            else:
                print(f"HTTP error occurred: {http_err}")
                time.sleep(1)
        except Exception as err:
            print(f"Generic error: {err}")
            renewTor()
            connectTor()
            restart = False
            time.sleep(1)
    
    transactions = data['txs']
    n_transactions = data['n_tx'] 
    tot_rec_btc = to_btc(data['total_received']) 
    tot_sent_btc = to_btc(data['total_sent'])
    actual_balance = to_btc(tot_rec_btc - tot_sent_btc)
    
    n_in_tx = 0 
    n_out_tx = 0 
    min_sent_btc = -1  
    Max_sent_btc = 0.0   
    min_rec_btc = -1  
    Max_rec_btc = 0     
    min_tx_interval = -1  
    Max_tx_interval = 0   
    Max_fee = 0  
    min_fee = -1  
    Max_in_tx_size = 0   
    min_in_tx_size = -1  
    Max_out_tx_size = 0  
    min_out_tx_size = -1  
    
    
    ins = []
    outs = []
    intervals = []
    fees = []
    tot_fees = 0
    tot_interval = 0
    curr_time  = 0
    prev_time = 0
    tot_tx_in = 0
    tot_tx_out = 0
    n_reused_address_in = 0
    n_reused_address_out = 0
    
    date_count = defaultdict(int)
    
    for tx in transactions:
        #Temporal distribution of transactions
        curr_time = int(tx['time'])
        date_only = datetime.utcfromtimestamp(curr_time).date()
        date_count[date_only] += 1
        
        if prev_time != 0:
            curr_interval = abs(prev_time - curr_time)
            intervals.append(curr_interval)
            tot_interval += curr_interval
            if(curr_interval > Max_tx_interval):
                Max_tx_interval = curr_interval
            if((curr_interval < min_tx_interval) | (min_tx_interval < 0)):
                min_tx_interval = curr_interval
        prev_time = curr_time
    
        #Value distribution of transactions
        val = to_btc(tx['result'])
        if val >= 0: #Incoming transaction
            n_in_tx += 1
            ins.append(val)
            if val > Max_rec_btc:
                Max_rec_btc = val
            if ((min_rec_btc < 0) | (val < min_rec_btc)):
                min_rec_btc = val
        else: #Outgoing transaction
            n_out_tx += 1
            val = abs(val)
            outs.append(val)
            if (val > Max_sent_btc):
                Max_sent_btc = val
            if ((min_sent_btc < 0) | (val < min_sent_btc)):
                min_sent_btc = val
        #Fees        
        fee = to_btc(tx['fee'])
        tot_fees += fee
        fees.append(fee)
        if(fee > Max_fee): Max_fee = fee
        if((min_fee < 0) | (fee < min_fee)): min_fee = fee
    
        #Reused addresses
        tx_ins = tx['inputs']
        tx_outs = tx['out']
        
        tot_reuse = 0
        for addr in tx_ins:
            try:
                if(addr['prev_out']['addr'] == address): tot_reuse += 1
            except KeyError as k:
                print(f"Unknown input address {address}")
        if (tot_reuse > 1): n_reused_address_in += 1
            
        tot_reuse = 0
        for addr in tx_outs:
            try:
                if(addr['addr'] == address): tot_reuse += 1
            except KeyError as k:
                print(f"Unknown output address {address}")
        if (tot_reuse > 1): n_reused_address_out += 1
    
        #Transaction size
        tx_ins_len = len(tx_ins)
        tx_outs_len = len(tx_outs)
        tot_tx_in += tx_ins_len
        tot_tx_out += tx_outs_len
        
        if (tx_ins_len > Max_in_tx_size): Max_in_tx_size = tx_ins_len
        if ((min_in_tx_size < 0) | ( tx_ins_len < min_in_tx_size)): min_in_tx_size = tx_ins_len
    
        if (tx_outs_len > Max_out_tx_size): Max_out_tx_size = tx_outs_len
        if ((min_out_tx_size < 0) | ( tx_outs_len < min_out_tx_size)): min_out_tx_size = tx_outs_len

    avg_sent_btc = 0
    avg_rec_btc = 0
    if n_out_tx != 0:
        avg_sent_btc = tot_sent_btc / n_out_tx
    if n_in_tx != 0:
        avg_rec_btc = tot_rec_btc / n_in_tx
        
    if(len(outs) > 1):
        std_dev_sent_btc = statistics.stdev(outs)
    else:
        std_dev_sent_btc = 0

    if(len(ins) > 1):
        std_dev_rec_btc = statistics.stdev(ins)
    else:
        std_dev_rec_btc = 0
        
    avg_tx_interval = tot_interval / n_transactions

    if(len(intervals) > 1):
        std_dev_tx_interval = statistics.stdev(intervals)
    else:
        std_dev_tx_interval = 0
        
    Max_daily_tx = max(date_count.values())  
    min_daily_tx = min(date_count.values())  
    avg_daily_tx = sum(date_count.values()) / len(date_count)
    if(len(date_count.values()) > 1):
        std_dev_daily_tx = statistics.stdev(date_count.values())
    else:
        std_dev_daily_tx = 0
        
    avg_fee = tot_fees / n_transactions

    if(len(fees) > 1):
        std_dev_fee = statistics.stdev(fees)
    else:
        std_dev_fee = 0
        
    avg_in_tx_size = tot_tx_in / n_transactions    
    avg_out_tx_size = tot_tx_out / n_transactions  
    tx_ratio = 0
    if n_out_tx != 0:
        tx_ratio = n_in_tx / n_out_tx  

    if min_tx_interval < 0:
        min_tx_interval = 0
    if min_sent_btc < 0:
        min_sent_btc = 0
    if min_rec_btc < 0:
        min_rec_btc = 0
    
    del ins
    del outs
    del intervals
    del fees
    del date_count
    gc.collect()
    eps = 1
    yield pd.DataFrame({
            'address': [address],
            'n_transactions': [np.log(n_transactions + eps)],
            'n_in_tx': [np.log(n_in_tx + eps)],   
            'n_out_tx': [np.log(n_out_tx + eps)],   
            'tx_ratio': [tx_ratio],   
            'Max_in_tx_size': [np.log(Max_in_tx_size + eps)],
            'min_in_tx_size': [np.log(min_in_tx_size + eps)],  
            'avg_in_tx_size': [np.log(avg_in_tx_size + eps)],  
            'Max_out_tx_size': [np.log(Max_out_tx_size + eps)],  
            'min_out_tx_size': [np.log(min_out_tx_size + eps)],  
            'avg_out_tx_size': [np.log(avg_out_tx_size + eps)],  
            'min_tx_interval': [min_tx_interval],  
            'Max_tx_interval': [Max_tx_interval],  
            'avg_tx_interval': [avg_tx_interval],  
            'std_dev_tx_interval': [std_dev_tx_interval],   
            'tot_rec_btc': [tot_rec_btc],  
            'tot_sent_btc': [tot_sent_btc],      
            'min_sent_btc': [min_sent_btc],  
            'Max_sent_btc': [Max_sent_btc],   
            'avg_sent_btc': [avg_sent_btc],   
            'std_dev_sent_btc': [std_dev_sent_btc],  
            'min_rec_btc': [min_rec_btc],   
            'Max_rec_btc': [Max_rec_btc],   
            'avg_rec_btc': [avg_rec_btc],   
            'std_dev_rec_btc': [std_dev_rec_btc],  
            'min_daily_tx': [np.log(min_daily_tx + eps)],  
            'Max_daily_tx': [np.log(Max_daily_tx + eps)],  
            'avg_daily_tx': [np.log(avg_daily_tx + eps)],  
            'std_dev_daily_tx': [std_dev_daily_tx],  
            'Max_fee': [Max_fee], 
            'min_fee': [min_fee],  
            'avg_fee': [avg_fee],  
            'std_dev_fee': [std_dev_fee], 
            'n_reused_address_in': [np.log(n_reused_address_in + eps)],  
            'n_reused_address_out': [np.log(n_reused_address_out + eps)]})