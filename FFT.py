import math

# Lookup tables. Only need to recompute when size of FFT changes


class FFT:
    def __init__(self, n):
        self.n = n
        self.m = int(math.log(self.n) / math.log(2))

        # Make sure n is a power of 2
        if self.n != 1 << self.m:
            raise ValueError("FFT length must be a power of 2")

        # Precompute tables
        self.cos = [0]*int(self.n / 2)
        self.sin = [0]*int(self.n / 2)

        for i in range(0, int(self.n / 2)):
            self.cos[i] = math.cos(-2 * math.pi * i / self.n)
            self.sin[i] = math.sin(-2 * math.pi * i / self.n)

    # ***************************************************************
    #          * fft.c
    #          * Douglas L. Jones
    #          * University of Illinois at Urbana-Champaign
    #          * January 19, 1992
    #          * http://cnx.rice.edu/content/m12016/latest/
    #          *
    #          *   fft: in-place radix-2 DIT DFT of a complex input
    #          *
    #          *   input:
    #          * n: length of FFT: must be a power of two
    #          * m: n = 2**m
    #          *   input/output
    #          * x: double array of length n with real part of data
    #          * y: double array of length n with imag part of data
    #          *
    #          *   Permission to copy and use this program is granted
    #          *   as long as this header is included.
    #          ****************************************************************

    def fft(self, re, im):
        # Bit reverse
        j = 0
        n2 = int(self.n / 2)
        for i in range(1, self.n - 1):
            n1 = n2
            while j >= n1:
                j = j - n1
                n1 = int(n1 / 2)
            j = j + n1

            if i < j:
                t1 = re[i]
                re[i] = re[j]
                re[j] = t1
                t1 = im[i]
                im[i] = im[j]
                im[j] = t1
        # FFT
        n2 = 1

        for i in range(0, self.m):
            n1 = n2
            n2 = n2 + n2
            a = 0

            for j in range(0, n1):
                c = self.cos[a]
                s = self.sin[a]
                a += 1 << (self.m - i - 1)

                for k in range(j, self.n, n2):
                    t1 = c * re[k + n1] - s * im[k + n1]
                    t2 = s * re[k + n1] + c * im[k + n1]
                    re[k + n1] = re[k] - t1
                    im[k + n1] = im[k] - t2
                    re[k] = re[k] + t1
                    im[k] = im[k] + t2
