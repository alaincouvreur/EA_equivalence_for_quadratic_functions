import time
load("Equivalence.py")

verbose = True

# Testing a function from n bits to m bits
m = 6
n = 6

# If you wish to test equivalent function or not
equiv = True   


# Number of successive tests you wish to perform
ntests = 1
if verbose:
    assert(ntests == 1),\
        "CAUTION: in verbose mode, only one test can be launched!"

# Counter to estimate the average number of guesses
ntries = 0

# How many pairs (v_i, w_i) do you try to guess simultaneously?
s = 3

# If the solution space of the linear system has dimension
# below this threshold, then perform exhaustive search in
# the solution space.
threshold = 10


#----------------------------------------------------------------#

print "-----------------"
print "Testing the equivalence of pairs of functions from "\
    + str(n) + " bits to " + str(m) + " bits.\n-----------------"

for i in range(ntests):
    timings = []
    print "\n**** Test #" + str(i+1) + " ****"
    F = random_vector_function(m, n, 2)

    # Generate function G, either equivalent to F or random.
    if equiv:
        A = random_invertible_matrix(m)
        B = random_invertible_matrix(n)
        C = random_full_rank_matrix(m, n)
        a = random_matrix(GF(2), m, 1)
        b = random_matrix(GF(2), n, 1)
        G = EAE_function(F, A, a, B, b, C)
    else:
        G = random_vector_function(m, n, 2) 

    start = time.time()    
    Z = get_equivalence(F, G, threshold, s, verbose)
    end = time.time()
    timings.append(end-start)
    print "Time : " + str(end-start) + " seconds."
    
    ntries += Z[5]
    if verbose:
        ranks = Z[6]
        lin_ranks = [r[0] for r in ranks]
        s1 = set(lin_ranks)
        lin_ranks = list(s1)
        aff_ranks = [r[1] for r in ranks]
        s2 = set(aff_ranks)
        aff_ranks = list(s2)
        lin_ranks.sort()
        aff_ranks.sort()
        #print "\n--------------------"
        #print "Observed ranks of the linear and affine systems of the "+\
        #  "various guesses"
        #print "Linear systems : " + str(lin_ranks)
        #print "Affine systems : " + str(aff_ranks)
        #print "--------------------\n"

print "Average number of tries = " + str(ntries * 1. / ntests)
print "Average time = " + str(sum(timings) * 1. / len(timings)) + " seconds."
