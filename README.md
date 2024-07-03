# OTFS-ISAC
IEEE/CIC ICCC 2024 Demo Proposal: OTFS-based ISAC System
## Introduction
The orthogonal time-frequency space (OTFS) technique is an innovative modulation scheme that represents wireless channels and modulates data in the delay-Doppler domain. This demo presents an integrated sensing and communication (ISAC) system based on OTFS signaling through a software-defined radio (SDR) platform. The OTFS-ISAC supports simultaneous communication and sensing in real-time, it also has time domain signal recording and delay-Doppler channel sounding functions. As a validation platform based on an open source framework, OTFS-ISAC provides a universal signal transmitting and acquisition workflow. Therefore, further studies can be conducted by modifying the respective system block. 



## Main features

**1. OTFS-assisted ISAC:** The OTFS-ISAC can work in communication and sensing modes simultaneously. Both modes use the same waveform, two subsystems can be turned on/off depending on the user setting.

**2. Streaming and channel sounding:** OTFS-ISAC has a signal recording function and supports simultaneous transmission and acquisition. As part of the system block, the effective DD domain channel matrix can be extracted after channel estimation.

**3. Open source:** We build the OTFS-ISAC based on the USRP Hardware Driver (UHD) framework, which is an open-source software with a large developer community. The codes are $100\%$ written in Python and easy to deploy in Linux, Windows and Mac OS with a wide range of Software Defined Radio (SDR) devices supported.

## System workflow
<p align = "center"> 
<img src=".\figures\workflow.png " width = "400" />
</p > 
 
<p align = "center"> 
<img src=".\figures\frame.png " width = "400" />
</p > 

## System setup
<p align = "center"> 
<img src=".\figures\devices.png " width = "400" />
</p > 

## License
**OTFS-ISAC** is covered under the MIT licensed.

 