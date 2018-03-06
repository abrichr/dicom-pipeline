Part 1: Parse the DICOM images and Contour Files

1) I verified that I was parsing the contours correctly by plotting the resulting mask over the corresponding DICOM and visually confirming that they matched (i.e. that the mask completely covered and was completely contained within a single structure of relatively uniform brightness). 

2) parsing.py was modified to `import pydicom as dicom` (instead of `import dicom`) since the package was renamed as of version 1.0 (see https://pydicom.github.io/pydicom/stable/transition_to_pydicom1.html).

3) If the pipeline were to be run on millions of images, and speed was paramount, I would parallelize it to run as fast as possible by having the main process create a `multiprocessing.Pool` of processes, each responsible for reading dicom/contour pairs. The parent process would then be responsible for feeding the child processes with the appropriate paths, and yielding the resulting data tuples back to the caller.

If additional scale were required, I would distribute the child processes on multiple machines. The parent process would yield locators to the child jobs, which would in turn yield the data directly to the caller.

4) If this pipeline were parallelized, I would add additional logic to ensure that each dicom/contour pair is only ever read by a single child process.
