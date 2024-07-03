import threading
from  multiprocessing import JoinableQueue,Process
import uhd
import numpy as np
import matplotlib.pyplot as plt
import time
import mat73
import scipy 
import logging
from logger import Log
import sys

def OTFS_demodulation(r,M,N):
    r_mat = r.reshape(N,M) #python numpy库中reshape是按照行进行排列的,在matlab中reshape是将数据按照列进行
    r_mat = np.transpose(r_mat)
    Y = np.fft.fft(r_mat, axis=0)/np.sqrt(M)#Numpy的fft.fft默认返回每行的结果，Matlab的fft默认返回每列的结果。
    Y = np.transpose(Y)
    y = np.transpose(np.fft.ifft(np.transpose(np.fft.fft(Y, axis=0)), axis=0))/np.sqrt(N/M)
    return y

def rx(rxStreamer,rxSamplesLen,usrp,event):
    # Receive Samples
    samples = np.zeros(rxSamplesLen, dtype=np.complex64)
    event.set()
    rxTimeHost = np.array(time.time())
    rxTimePPS = np.array(usrp.get_time_last_pps().get_real_secs())
    # print("接收起始时间 PPS:%.3f Host:%.3f"%(rxTimePPS, rxTimeHost))
    # log.debug("接收起始时间 PPS:%.6f Host:%.6f"%(rxTimePPS, rxTimeHost))
    for i in range(rxSamplesLen//1000):
        rxStreamer.recv(recv_buffer, metadata)
        samples[i*1000:(i+1)*1000] = recv_buffer[0]
    return samples,rxTimeHost
    

def tx(txStreamer,txData,usrp,event):
    buffer_samps = txStreamer.get_max_num_samps()
    # print(buffer_samps)
    txData_len = txData.shape[-1]
    if txData_len < buffer_samps:
        txData = np.tile(txData,
                                    (1, int(np.ceil(float(buffer_samps)/txData_len))))
        txData_len = txData.shape[-1]
    send_samps = 0
    max_samps = int(np.floor(duration * sample_rate))
    if len(txData.shape) == 1:
        txData = txData.reshape(1, txData.size)
    if txData.shape[0] < len(st_args.channels):
        txData = np.tile(txData[0], (len(st_args.channels), 1))
    # Now stream
    # metadataTx = uhd.types.TXMetadata()
    # if start_time is not None:
    #     metadataTx.time_spec = start_time
    event.wait()
    txTimeHost = np.array(time.time())
    txTimePPS = np.array(usrp.get_time_last_pps().get_real_secs())
    # log.debug("发射起始时间 PPS:%.6f Host:%.6f"%(txTimePPS, txTimeHost))
    while send_samps < max_samps:
        real_samps = min(txData_len, max_samps-send_samps)
        if real_samps < txData_len:
            samples = txStreamer.send(txData[:, :real_samps], metadataTx)
        else:
            samples = txStreamer.send(txData, metadataTx)
        send_samps += samples


    return txData,txTimeHost

def txrx(loops,txData,txStreamer,rxStreamer,usrp,rxTimeQueue,txTimeQueue,rxQueue):
    # rx Streamer setting
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rxStreamer.issue_stream_cmd(stream_cmd)
    # time.sleep(0.03)
    rxSamplesLen = num_samps + int(sample_rate*0.0)

    event = threading.Event()

    for i in range(loops):
        thread1 = NewThread(target=tx,args=(txStreamer,txData,usrp,event))
        thread2 = NewThread(target=rx,args=(rxStreamer,rxSamplesLen,usrp,event))
        # thread1 = NewThread(target=tx,args=(num_samps,))
        thread1.start()
        thread2.start()


        [txSamples,txTimeHost] = thread1.join() 
        [rxSamples,rxTimeHost] = thread2.join()
        rxQueue.put(rxSamples)
        rxTimeQueue.put(rxTimeHost)
        txTimeQueue.put(txTimeHost)

        # rxTimeHost = rxTimeQueue.get()
        # txTimeHost = txTimeQueue.get()
        # rxSamples = rxQueue.get()  
        # rxTimeQueue.task_done()
        # txTimeQueue.task_done()
        # rxQueue.task_done()
        # print(rxQueue.qsize())
        log.info("队列剩余处理发射轮次：%.0f"%rxQueue.qsize())
        # print(i)

    # Stop rx Streamer
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rxStreamer.issue_stream_cmd(stream_cmd)
    rxStreamer = None

    # Send EOB to terminate Tx
    # metadataTx.end_of_burst = True
    # txStreamer.send(np.zeros((len(st_args.channels), 1), dtype=np.complex64), metadataTx)
    # Help the garbage collection
    txStreamer = None


    # print('全部帧已发送')
    log.debug("所有数据已收发，等待计算中")
        # return txSamples,txTimeHost,rxSamples,rxTimeHost


def clearQueue(q):
    while not q.empty():
        q.get()
        q.task_done()
        time.sleep(0.001)
        # print(q.qsize())
    # q.join()

##   u代表原矩阵，shiftnum1代表行，shiftnum2代表列。
def circshift(u,shiftnum1,shiftnum2):
    h,w = u.shape
    if shiftnum1 < 0:
        u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
    else:
        u = np.vstack((u[(h-shiftnum1):,:],u[:(h-shiftnum1),:]))
    if shiftnum2 > 0:
        u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
    else:
        u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))
    return u

def findMax(y):
    max_index = np.unravel_index(np.argmax(np.abs(y), axis=None), np.abs(y).shape)
    max_value = y[max_index]
    return max_value,max_index


class NewThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
    def run(self):
        if self._target != None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return
 
if __name__ == '__main__':
    # plt.switch_backend('Agg')
    ## 加载matlab数据文件，读取变量
    log = Log('MyLog')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    log.changeLevel(logging.DEBUG)
    log.info('---OTFS ISAC Real Time System---')
    path = r"./2024_1_16_sigGen10Smallguard256128CP.mat" # 在这里改你文件路径
    data = mat73.loadmat(path)

    sync = 0 #显示Sync
    comm = 1 #显示Constellation
    sensing = 0 #显示Doppler

    # 在这里选择你mat文件的变量
    txData = data['txData'].astype(np.complex64)
    # txData = txData.astype(np.complex64)
    preamble = data['preamble']
    M = int(data['M'])
    N = int(data['N'])
    kv = int(data['kv'])
    ltau = int(data['ltau'])
    Nzp = 50
    Ncp = data['Ncp']
    ltaumax = 1
    kvmax = 2
    log.info('---Waveform and Parameters Loaded---')

    log.info('Starting USRP')
    ## 设置USRP发射接受参数
    usrp = uhd.usrp.MultiUSRP()
    # RX side setting
    # select Dboard (mboard slot A or B, dboard name: 0)
    subdev_spec = uhd.usrp.SubdevSpec("B:0") #Ettus please go fuck yourself your python doc just like shit
    # print(subdev_spec)
    log.info(subdev_spec)

    usrp.set_rx_subdev_spec(subdev_spec)
    # Select antenna (TX/RX or RX2)
    usrp.set_rx_antenna('TX/RX')

    # TX side setting
    subdev_spec = uhd.usrp.SubdevSpec("A:0")  
    # print(subdev_spec)
    log.info(subdev_spec)
    usrp.set_tx_subdev_spec(subdev_spec)

    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")

    loops = 10
    # duration = 1
    center_freq = 5.6e9 # Hz
    sample_rate = 1e6 # Hz
    num_samps = len(txData)
    duration = num_samps/sample_rate

    rxGain = 20 # dB
    txGain = 10

    usrp.set_rx_rate(sample_rate, 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
    usrp.set_rx_gain(rxGain, 0)

    usrp.set_tx_rate(sample_rate, 0)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
    usrp.set_tx_gain(txGain, 0)

    # Set up the rx stream and receive buffer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    metadata = uhd.types.RXMetadata()
    rxStreamer = usrp.get_rx_stream(st_args)
    recv_buffer = np.zeros((1, 1000), dtype=np.complex64)

    # Set up the tx stream and receive buffer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    metadataTx = uhd.types.TXMetadata()
    txStreamer = usrp.get_tx_stream(st_args)
    tran_buffer = np.zeros((1, 1000), dtype=np.complex64)

    ##生产者消费者模型，队列缓冲
    log.info("建立队列线程间通信")
    rxTimeQueue,txTimeQueue,rxQueue = JoinableQueue(),JoinableQueue(),JoinableQueue()
    
    p1 = threading.Thread(target=txrx,args=(loops,txData,txStreamer,rxStreamer,usrp,rxTimeQueue,txTimeQueue,rxQueue))
    log.info("启动USRP同步收发线程")
    p1.start()

    Doppler_waterfall = np.zeros((4*kv+1,1))
    while True:
        
        rxTimeHost = rxTimeQueue.get()
        txTimeHost = txTimeQueue.get()
        rxSamples = rxQueue.get()
        log.info("队列剩余处理发射轮次：%.0f"%rxQueue.qsize())
        # log.info("Block Size: %.1f byte"%sys.getsizeof(rxSamples)/2**10)
        # log.debug(sys.getsizeof(rxSamples)/2**10)
        # log.debug(rxSamples.dtype)
        # log.debug(rxSamples.shape)


        
        lagSamplesNum = int(np.abs(txTimeHost-rxTimeHost)*sample_rate)
        # print(lagSamplesNum)
        # rxSamplesCut = rxSamples[lagSamplesNum:lagSamplesNum+num_samps-1]
        rxSamplesCut = rxSamples

        # Calculate cross correlation
        detmet = np.abs(np.correlate(rxSamplesCut,preamble,mode='full'))
        detmet = np.divide(detmet,np.max(detmet))

        # 滑动窗口大小
        window_size = 1000
        # move average (bad method, may false alarm)
        baseline = 16*np.convolve(detmet, np.ones(window_size)/window_size, mode='valid')
        baseline = np.append(baseline,np.ones(min(len(detmet),len(np.ones(window_size)))-1))

        locs_f = np.array(np.where(detmet>baseline)).squeeze()
        

        # Scipy find_peaks method (works well)        
        locs,peak_property = scipy.signal.find_peaks(detmet,height = 0.2 ,distance=0.9e5)
        # Discard tail 
        locs = locs[0:10]
        # locs = np.array(locs).reshape(1,len(locs))
        
        # print(np.shape(locs_f))
        # print(np.shape(locs))
        # print(np.shape(baseline))
        # print(np.shape(detmet))
        
        
        if sync:
            plt.figure(1)
            plt.clf()
            plt.subplot(211)
            plt.plot(np.real(rxSamplesCut),label='Recieved signal')
            plt.legend()
            plt.title('Received time domain signal')
            plt.subplot(212)
            plt.plot(detmet,label='Cross correlation')
            # plt.plot(threshold)
            plt.plot(locs_f,detmet[locs_f],'bv',label='False alarm')
            plt.plot(locs,detmet[locs],'ro',markersize=10,label='Truth')
            plt.legend()
            plt.title('Preamble detection')
            plt.pause(0.001)
            # plt.show()
        
        for blockNo in range(len(locs)-0): 
            i = 1
            for frameNo in range(1,3+1):
                if blockNo == 9 and frameNo==3:
                    log.debug("skip the tail")
                    continue
                # print(frameNo)
                # print(locs[0,blockNo])
                startPoint = int(locs[blockNo]+i+(Nzp+Ncp)*frameNo+M*N*(frameNo-1))
                endPoint = int(locs[blockNo]+i+(Nzp+Ncp)*frameNo+M*N*frameNo)
                dataBlock = rxSamplesCut[startPoint:endPoint]
                # print(dataBlock.shape)
                # print(M*N)
                try:
                    y = OTFS_demodulation(dataBlock,M,N)
                except Exception as e:
                    log.warning('Tail alarm')
                    log.error(e)
                    continue
                # print(y.shape)
                # dataBlock = txData[0:100]

                [max_value,max_index] = findMax(y)

                pilot_region = y[max_index[0]-2*kv:max_index[0]+2*kv+1,max_index[1]-ltau:max_index[1]+ltau+1]
                try:
                    Doppler_tap = np.abs(y[max_index[0]-2*kv:max_index[0]+2*kv+1,max_index[1]]).reshape((4*kv+1,1))
                except Exception as e:
                    log.critical("Bad sync")
                    log.critical(e)
                    continue

                # print(np.shape(Doppler_tap))
                # print(np.shape(Doppler_waterfall))
                Doppler_waterfall = np.append(Doppler_waterfall,Doppler_tap,axis=1)
                if np.shape(Doppler_waterfall)[1]>100:
                    Doppler_waterfall = np.delete(Doppler_waterfall,0,axis=1)

                if comm:
                    # channel estimation
                    estimate_region = y[max_index[0]-2*kv+kvmax:max_index[0]+2*kv-kvmax+1,max_index[1]-ltau+ltaumax:max_index[1]+ltau-ltaumax+1]
                    [max_value,max_index] = findMax(estimate_region)
                
                    Est_h = circshift(estimate_region,max_index[0],-max_index[1]-1)

                    # extend the effective channel matrix
                    Est_h[np.abs(Est_h)<0.1*np.abs(max_value)] = 0 # thresholding
                    Est_h_Large = np.zeros(np.shape(y),dtype=np.complex64)
                    Est_h_Large[0:round(np.shape(Est_h)[0]/2), 0:round(np.shape(Est_h)[1]/2)] = Est_h[0:round(np.shape(Est_h)[0]/2), 0:round(np.shape(Est_h)[1]/2)]  # Top left
                    Est_h_Large[0:round(np.shape(Est_h)[0]/2), -round(np.shape(Est_h)[1])+round(np.shape(Est_h)[1]/2):] = Est_h[0:round(np.shape(Est_h)[0]/2), round(np.shape(Est_h)[1]/2):]  # Top right
                    Est_h_Large[-round(np.shape(Est_h)[0])+round(np.shape(Est_h)[0]/2):, 0:round(np.shape(Est_h)[1]/2)] = Est_h[round(np.shape(Est_h)[0]/2):, 0:round(np.shape(Est_h)[1]/2)]  # Bottom left
                    Est_h_Large[-round(np.shape(Est_h)[0])+round(np.shape(Est_h)[0]/2):, -round(np.shape(Est_h)[1])+round(np.shape(Est_h)[1]/2):] = Est_h[round(np.shape(Est_h)[0]/2):, round(np.shape(Est_h)[1]/2):]  # Bottom right
                    Est_h_Large = Est_h_Large
                    sigma = 1e-2
                    temp1 = np.conj(np.fft.fft2(Est_h_Large))
                    temp2 = np.square(np.abs(np.fft.fft2(Est_h_Large)))+sigma
                    temp = np.divide(temp1,temp2)
                    x_est = np.fft.ifft2(np.fft.fft2(y)*temp)*N

                    [max_value,max_index] = findMax(x_est)
                    # print(max_index)
                    plot_num = 12
                    if frameNo==1:
                        plt.figure(4)
                        plt.clf()
                        plt.subplot(121)
                        plt.plot(np.real(y[2:plot_num,2:plot_num]),np.imag(y[2:plot_num,2:plot_num]),'ro-',markersize=10,linewidth=1)
                        plt.title('Received constellation')               
                        plt.subplot(122)
                        plt.plot(np.real(x_est[2:plot_num,2:plot_num]),np.imag(x_est[2:plot_num,2:plot_num]),'ro-',markersize=10,linewidth=1)
                        # plt.title([blockNo+1,frameNo])
                        plt.title('Equalized constellation')
                        plt.ioff()
                        plt.pause(0.001)
 
                if sensing&frameNo==1:
                    plt.figure(3)
                    plt.clf()
                    ax, fig = plt.gca(), plt.gcf()
                    # Z = np.abs(y)
                    # Z = np.abs(Est_h)
                    # Z = np.abs(x_est)
                    # Z = np.abs(pilot_region)
                    Z = Doppler_waterfall
                    im = ax.pcolormesh(Z, 					# x和y可以不给出                        
                        cmap = plt.get_cmap('jet'), 	# cmap颜色系统
                        alpha = 0.8, 				# 透明度
                        edgecolors = None # 每个色块边缘的颜色，可选'none'、None、'face'、color、color sequence五种，分别指示无颜色、内置基础色、临近格子颜色或是自定义颜色（序列）
                    )
                    fig.colorbar(im, ax = ax)
                    # im = ax.plot(np.real(y),np.imag(y),'ro')
                    # im = ax.plot(np.real(x_est[0:20,0:20]),np.imag(x_est[0:20,0:20]),'ro')
                    plt.title('Doppler shift')
                    plt.ioff()
                    plt.pause(0.001)
                    # # plt.show()



        rxTimeQueue.task_done()
        txTimeQueue.task_done()
        rxQueue.task_done()
        if rxQueue.empty():
            break

    p1.join()

    clearQueue(rxTimeQueue)
    clearQueue(txTimeQueue)
    clearQueue(rxQueue)
    log.info("主进程结束")
    plt.show()

        

 