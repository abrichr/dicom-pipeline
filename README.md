## Part 1: Parse the DICOM images and Contour Files

1) I verified that I was parsing the contours correctly by plotting the resulting mask over the corresponding DICOM and visually confirming that they matched (i.e. that the mask completely covered and was completely contained within a single structure of relatively uniform brightness). 

![Example Output](/tmp_0007.png)

2) parsing.py was modified to `import pydicom as dicom` (instead of `import dicom`) since the package was renamed as of version 1.0 (see https://pydicom.github.io/pydicom/stable/transition_to_pydicom1.html).

3) If the pipeline were to be run on millions of images, and speed was paramount, I would parallelize it to run as fast as possible by having the main process create a `multiprocessing.Pool` of processes, each responsible for reading dicom/contour pairs. The parent process would then be responsible for feeding the child processes with the appropriate paths, and yielding the resulting data tuples back to the caller.

    If additional scale were required, I would distribute the child processes on multiple machines. The parent process would yield locators to the child jobs, which would in turn yield the data directly to the caller.

4) If this pipeline were parallelized, I would add additional logic to ensure that each dicom/contour pair is only ever read by a single child process, and to prevent the pipeline from running out of memory and/or loading too much data into memory than is required at a time.

## Part 2: Model training pipeline

1) Loading batches asynchronously is done via an object of class BatchFeeder, which loads batches in a separate process. Upon instantiation, the BatchFeeder pre-emptively loads the first batch. Then, once a batch is requested, it loads the next batch before returning the one that was previously loaded. This way a batch is returned as soon as possible after it is requested, but without wasting memory by loading more than one at a time.

    I also considered loading batches asynchronously via a multiprocessing.Pool. However, Pool map functions consume their entire iterable as soon as possible, thereby wasting memory. 

    I used multiprocessing instead of multithreading because the former avoids Python's Global Interpreter Lock, thereby fully leveraging multiple processors.

2) The primary safeguard that prevents the pipeline from crashing when run on thousands of studies is the fact that it only loads a single batch at a time. This prevents it from using too much memory unnecessarily.

3) I modified the function from Part 1 by moving the functionality that loads the file contents into a different function which accepts an iterator over path tuples. This way I was able to load subsets of the data in batches, instead of all of it at once.

4) To verify that the pipeline was working correctly, I observed the logs to confirm that only a single batch was being read at a time, and that it was being read during the time at which a model would be training on it.

    I also performed the same visual inspection as in Question 1 of Part 1.

5) TODO:

- Use a configurable number of background process to load batches more quickly. Currently only one processed is used.
- Make the number of batches to load in the background at a time configurable. Currently only one batch is loaded at a time.
- More thorough unit test coverage. 
- Log to cloud logging service
- Deploy across multiple servers
