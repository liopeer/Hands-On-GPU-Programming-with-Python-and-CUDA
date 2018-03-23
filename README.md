# Hands-On-GPU-Programming-with-Python-and-CUDA
Hands-On GPU Programming with Python and CUDA, published by Packt

## Hardware and Software Requirements
In this text, we assume that you have a Maxwell (2014-era) or Pascal (2016-era) Nvidia GPU, or later; you should have at least have an entry-level Maxwell GTX 750 or Pascal GTX 1050, or the equivalent (e.g., a GTX 860M if you are using a laptop).  Generally speaking, you will be able to work through this book with almost any entry-level gaming PC released in 2014 or later that has an Nvidia GPU (desktop or laptop). 

Both the Windows 10 and Linux Operating Systems provide suitable environments for CUDA programming.  (Windows 10 is an entirely suitable choice for laptop users for beginning GPU programming, due to the relative ease of the installation of the Nvidia drivers and CUDA environment compared to Linux.)  I would urge Linux users to consider using a Long Term Support (LTS) Ubuntu Linux distribution (16.04 or 18.04) or any LTS Ubuntu derivatives (e.g., Lubuntu, Xubuntu, Ubuntu Mate, Linux Mint), due to the strong support these distributions receive from Nvidia;  in particular, I am using Linux Mint 18.3 on one of my systems at the time of writing, and I find it works very well for a CUDA environment.

While we will go over particular development environments in the following chapter, I suggest the Anaconda Python 2.7 distribution (available at https://www.anaconda.com/download/ ).  In particular, I will be using this Python distribution throughout the text for the examples I will be giving.  Anaconda Python is available for both Windows and Linux, it is very easy to install, and it contains a large number of optimized mathematical, machine learning, and data science related libraries that will come in useful, as well as some very nice pre-packaged Integrated Development Environments (IDE) such as Spyder and Jupyter.  Moreover, Anaconda can be installed on a user-by-user basis, and provides an isolated environment from the system installation of Python.  For these reasons, I suggest you start with the Anaconda Python distribution.

While we will discuss the particular required compilers and development environment in the subsequent chapter, it should be noted that any version of the CUDA Toolkit from 8.0 onwards will work for any of the examples in this book.  I am currently using CUDA 9.1 on my Windows 10 system, while I am still on CUDA 8.0 on my Linux system, and both work very well.  For Windows, I would suggest Visual Studio Community Edition 2015, due to its tight integration with Anaconda;  for Linux, a standard gcc installation along with the Eclipse IDE for C++ from your distribution’s repository should be enough.  (From the Ubuntu bash command line: “sudo apt-get update && sudo apt-get install build-essentials && sudo apt-get install eclipse-cdt” )
