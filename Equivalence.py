##########################################################################
##                            Vector Functions                          ##
##########################################################################

def random_boolean_function(n, degree):
    ## Generates a uniformly random Boolean function
    ## in n variables of bounded degree
    F = GF(2)
    R = PolynomialRing(GF(2), 'x', n)
    if n == 0 or degree == 0:
        return R(F.random_element())
    
    u = random_boolean_function(n-1, degree)
    v = random_boolean_function(n-1, degree-1)
    return R(u) + R.gens()[n-1] * R(v)



def random_vector_function(m, n, degree):
    ## returns an m x 1 matrix of functions
    ## in n variables of bounded degree
    return Matrix(m,1,[random_boolean_function(n, degree) for i in range(m)])



def random_invertible_matrix(n):
    Mat = MatrixSpace(GF(2), n)
    while True:
        A = Mat.random_element()
        if A.is_invertible():
            return A


def random_full_rank_matrix(m,n):
    r = min(m, n)
    Mat = MatrixSpace(GF(2), m, n)
    while True:
        A = Mat.random_element()
        if A.rank() == r:
            return A
        
    

        
##########################################################################
##              Construction of affine equivalent functions             ##
##########################################################################

def EAE_function(F, A, a, B, b, C):
    ## INPUTS : 
    ## - F a vector function with n inputs and m outputs
    ## - A an invertible m x m matrix, a an m x 1 matrix
    ## - B an invertible n x n matrix, b an n x 1 matrix
    ## - C a full rank m x n matrix
    ## 
    ## OUTPUTS : A*F(Bx + b) + Cx + a           

    ## 1. Some tests
    head_str = "[EAE_function] : "
    assert(F.ncols() == 1),\
        head_str + "input F should be a vector function"

    m = F.nrows()
    assert(m != 0), head_str + "input F should be a non empty function"
    assert(A.nrows() == A.ncols()),\
        head_str + "matrix A should be square"
    assert(B.nrows() == B.ncols()),\
        head_str + "matrix B should be square"
    assert(C.nrows() == A.ncols()),\
        head_str + "matrix C has not the good number of rows"
    assert(C.ncols() == B.nrows()),\
        head_str + "matrix C has not the good number of columns"
    
    R = F.base_ring()
    n = R.ngens()

    assert(A.nrows() == m),\
        head_str + "matrix A's size should be the output size of F"        
    assert(B.nrows() == n),\
        head_str + "matrix B's size should be number of variables of F"
    assert(C.ncols() == n),\
        head_str + "matrix C's size should be number of variables of F"
    assert(a.nrows() == m and a.ncols() == 1),\
        head_str + "a should be an m x 1 matrix"
    assert(b.nrows() == n and b.ncols() == 1),\
        head_str + "b should be an n x 1 matrix"
        
    
    ## 2. Construction of the equivalent function
    var = Matrix(R, n, 1, R.gens())
    variable_change = B * var + b
            
    ## 3. Construction of the new function
    G0 = F(variable_change.list())
    
    ## 4. Reduce modulo the field equations
    I = R.ideal([x**2 - x for x in R.gens()])
    for i in range(m):
        G0[i,0] = G0[i,0].mod(I)
            
    return A * G0 + C * var + a


def affine_equivalent_function(F, A, a, B, b):
    C = zero_matrix(GF(2), A.nrows(), B.nrows())
    return EAE_function(F, A, a, B, b, C)




########################################################################
##                     Some functions on matrices                     ##
########################################################################

def right_mult_mat(M):
    # Given a k x n matrix M.
    # returns the (kn) x k^2 matrix representing
    # the right multiplication by M,
    # i.e. the map Mat_{k x k} --> Mat_{k x n}
    #                   X      |-> XM
    k = M.nrows()
    I = identity_matrix(M.base_ring(), k)
    return I.tensor_product(M.transpose(), subdivide = False)


def left_mult_mat(M):
    # Given a k x n matrix M.
    # returns the (kn) x k^2 matrix representing
    # the left multiplication by M,
    # i.e. the map Mat_{n x n} --> Mat_{k x n}
    #                   X      |-> MX
    n = M.ncols()
    I = identity_matrix(M.base_ring(), n)
    return M.tensor_product(I, subdivide = False)


def rank_table(M):
    ## Given a k x n matrix M with entries
    ## in a polynomial ring in t variables,
    ## computes the rank table for any possible
    ## entry.
    ##
    ## For any i, the i-th entry of the table is the list
    ## of vectors v such that M(v) has rank i.
    assert(M.nrows() != 0 and M.ncols() != 0),\
        "[rank_table] : input matrix should be non empty"
            
    max_rank = min(M.nrows(), M.ncols())
    rank_table = [[] for i in range(max_rank + 1)]
    R = M.base_ring()
    
    for v in VectorSpace(GF(2), R.ngens()):
        r = M(v.list()).rank()
        rank_table[r].append(v)

    return rank_table


def rank_distribution(rt):
    ## Returns a table listing the cardinalities
    ## of the different entries of the rank table
    return [len(l) for l in rt]


def JacobianMatrix(F):
    head_str = "[JacobianMatrix] : "
    assert(F.ncols() == 1),\
        head_str + "The input function should be an m x 1 matrix"

    m = F.nrows()
    assert(m != 0),\
        head_str + "The input function should be non empty"

    R = F.base_ring()
    n = R.ngens()
    rows = [[F[j,0].derivative(xi) for xi in R.gens()] for j in range(m)]
    return Matrix(R, m, n, rows)


def JacLin(F):
    ## Computes the linear part
    ## of the Jacobian matrix.
    JF = JacobianMatrix(F)
    return JF - JF([GF(2)(0) for i in range(JF.ncols())])
    



###########################################################################
##               Main functions to detect equivalences                   ##
###########################################################################

def solve_system(JF, JG, vv, ww, verbose = False):
    ## This function is a single interation of the algorithm
    ## Computes (if exist) matrices A, B such that
    ## 1. A JG(vi) = JF(w_i) B for vi in vv; wi in ww
    ## 2. wi = Bvi for i \geq 2
    ##------
    ## Returns
    ##     - a solution space if non empty together with
    ##     - the rank lin_rank of the linear part of the system
    ##     - the rank aff_rank of the full affine system

    ## 1. Conditions
    head_str= "[solve_system] : "
    assert(JG.nrows() == JF.nrows()),\
        head_str + "JF and JG should have the same number of rows"
    assert(JF.ncols() == JG.ncols()),\
        head_str + "JF and JG should have the same number of columns"
    assert(len(vv) == len(ww)),\
        head_str + "vv and ww should have the same length"
    assert(len(vv) != 0),\
        head_str + "vv and ww should be nonempty"
        
    R = JF.base_ring()
    assert(JG.base_ring() == R),\
        head_str + "JF and JG should have the same base ring"

    m = JF.nrows()
    n = R.ngens()
    for v in vv:
        assert(len(v) == n),\
            head_str + "entries of vv are lists of length n"
        
    for w in ww:
        assert(len(w) == n),\
            head_str + "entries of ww are lists of length n"
    

    ## 2. Construction of the matrices we need            
    LJF = [left_mult_mat(JF(w.list())) for w in ww]
    RJG = [right_mult_mat(JG(v.list())) for v in vv]
    Rv = [right_mult_mat(Matrix(GF(2), n, 1, v.list())) for v in vv]

    List = [[RJG[i], -LJF[i]] for i in range(len(LJF))]
    List += [[zero_matrix(GF(2),n, m**2), r] for r in Rv]
    M = block_matrix(List)
    List2 = [0 for i in range(m * n * len(vv))]
    for w in ww:
          List2 += w.list()
          
    V = matrix(GF(2), (m + 1) * n * len(vv), 1, List2)

    ## 3. Solve the system
    assert(M.ncols() == m**2 + n**2),\
        head_str + "Number of variables should be n^2 + m^2 columns"

    lin_rank = M[:m*n*len(vv)][:].rank()
    aff_rank = M.rank()
    if verbose:
        print head_str + "nr of equations: " + str(M.nrows()) + "."
        print head_str + "nr of unknowns: " + str(M.ncols()) + "."
        print head_str + "Rank of the system : " + str(aff_rank) + "."
        print head_str +\
            "Rank of the system (without the affine equations): " +\
            str(lin_rank) + "."
    try:
        T = M.solve_right(V)
    except ValueError:
        if verbose:
            print head_str + "No solution."
        return (None, None, lin_rank, aff_rank)

    return (M.right_kernel(), T, lin_rank, aff_rank)




def AB_from_vector(vect, m, n):
    ## From a valid result of the linear system
    ## Reconstructs the pair (A, B) as a pair
    ## of matrices from a vector representation.
    assert(len(vect) == m*m + n*n),\
        "[AB_from_vector] : first entry should have length m^2+n^2"
    Am1 = matrix(GF(2), m, m, vect[ : m*m])
    B = matrix(GF(2), n, n, vect[m*m : ])
    return (Am1, B)
    


def reference(rank_tab, rank_dist, m, n):
    ## This function returns two lists
    ## - A list reference_vectors of n vectors in F_2^n
    ## - A list reference_ranks of n integers
    ## A reference vector v has an image
    ## Bv which should be easy to guess
    ## and the rank of JG(v) is stored in
    ## reference_ranks at the same position as v

    
    ## 1. Extracts a sorted distribution without 0 and without
    ##    double terms
    distrib = rank_dist[1:]
    dist_set = set(distrib)
    dist_set.remove(0)
    distrib = list(dist_set)
    distrib.sort()

    ## 2. Some variable
    max_rank = n-1 if (m == n) else min(m,n)
    reference_vectors = []
    reference_ranks = []

    ## 3. Go!
    for d in distrib:
        rank = max_rank


        ## 3.1. Finds the right most entry of cardinality d in the
        ## rank table
        while rank > 0:
            if rank_dist[rank] == d:
                for v in rank_tab[rank]:
                    # The new reference vector should be
                    # independent from the other ones
                    M = matrix(reference_vectors + [v])
                    if M.rank() == len(reference_vectors) + 1:
                        reference_vectors.append(v)
                        reference_ranks.append(rank)
                        if len(reference_vectors) >= n:
                            return (reference_vectors, reference_ranks)
            rank -= 1
    return None



def check_candidate(JF, JG, A, B):
    for v in basis(VectorSpace(GF(2), n)):
        if JG(v.list()) != A * JF((B * v).list()) * B:
            return False
    return True



def full_tuple(F,G,A,B):
    ## Returns the remainder of a tuple (a,b,C)
    ## from the knowledge of (A,B).
    ## The tuple is not unique. We choose the one with b = 0
    Zm = zero_matrix(GF(2), m, 1)
    b_cand = zero_matrix(GF(2), n, 1)
    G1 = affine_equivalent_function(F, A, Zm, B, b_cand)
    a_cand = (G - G1)([0 for i in range(n)])
    Cpoly = G-G1-a_cand
    C_cand = JacobianMatrix(Cpoly)
    return (a_cand, b_cand, C_cand)


def number_of_guesses(m, n, reference_ranks):
    ## Estimates a reasonable number of guesses s.
    ## If you don't use this function test with s = 2 or 3
    n_guesses = 0
    N = m**2 + n**2
    M = 0
    for r in reference_ranks:
        n_guesses += 1
        N -= r * (m+n-r)
        M += n - r
        if N <= M:
            return n_guesses
    return None



def get_equivalence(F, G, limit = 10, s = None, verbose = False):
    ## Inputs
    ##  - functions F, G
    ##  - limit is the maximum admissible dimension of
    ##    the solution set of a linear system
    ##    in which we perform brute force. By default it is set
    ##    to 10.
    ##  - The number s of vectors we have to guess
    ##     (suggested s = 2 or 3). By default, a value will be computed
    ##     using the function number_of_guesses
    ##  - Turn verbose to True if you wish to have a further details on the
    ##    running
    ##
    ## Extracts if exists a candidate pair (A,B) of
    ## for the linear parts of the
    ## affine equivalence

    ## 1. Jacobian matrices and verifications
    JF = JacLin(F)
    JG = JacLin(G)
    n = JG.ncols()
    m = JG.nrows()

    head_str = "[get_extended_affine_equivalence] : "
    assert(JF.ncols() == n),\
        head_str + "F, G should have the same number of variables"
    assert(JF.nrows() == m),\
        head_str + "F, G should have the same number of outputs"

    
    ## 2. Rank tables and distributions
    rank_tab1 = rank_table(JF)
    rank_tab2 = rank_table(JG)
    rank_dist1 = rank_distribution(rank_tab1)
    rank_dist2 = rank_distribution(rank_tab2)
    print "\n------------"
    print "Rank distributions : "
    print rank_dist1
    print rank_dist2
    print "------------"

    
    ## 3. Compares the rank distributions 
    if rank_dist1 != rank_dist2:
        print "F and G have not the same Jacobian rank distribution"
        print "They are not equivalent!\n"
        return (None, None, None, None, None, 0)
 

    ## 4. Reference vectors
    ref = reference(rank_tab2, rank_dist2, m, n) 
    ref_vec = ref[0]
    ref_rk = ref[1]

    if verbose:
        print "ref_rk " + str(ref_rk)

    ## 5. Evaluating the good number of guesses (unless it
    ## has already been defined)
    if s == None:
        s = number_of_guesses(m, n, ref_rk)
    print "number s of guesses = " + str(s)
    if verbose:
        print "Expected ranks : "
        rank_linear = sum([ref_rk[i] * (m+n-ref_rk[i])\
                           for i in range(s)])
        rank_linear = min(rank_linear, m**2+n**2)
        rank_affine = rank_linear + sum([n-ref_rk[i] for i in\
                                         range(s)])
        rank_affine = min(rank_affine, m**2+n**2)
        print "   - Linear system on (A, B) : " + str(rank_linear)
        print "   - Full affine system      : " + str(rank_affine) + "\n"
        raw_input("Press enter to continue...")
    
    ## 6. Enumeration strategy
    ## Here we determine the structure
    ## of the s-tuples of vectors we guess
    strategy = [[ref_rk[0], 1]]
    for i in range(1,s):
        if ref_rk[i] == ref_rk[i-1]:
            strategy[len(strategy)-1][1] += 1
        else:  
            strategy.append([ref_rk[i], 1])

    if verbose:
        print "strategy : " + str(strategy)

    iteration_arrangements = []
    for pair in strategy:
        l = len(rank_tab1[pair[0]])
        iteration_arrangements.append(Arrangements(range(l), pair[1]))

    ## 7. Iteration
    w_candidates = [VectorSpace(GF(2), n)(0) for i in range(s)]
    counter = 0
    rejection_counter = 0
    if verbose:
        ranks = []
    else:
        ranks = None
    
    for list_of_tuples in cartesian_product(iteration_arrangements):
        k = 0
        counter += 1

        ## 7.1. Construction of the tuple ww of candidate vectors
        for i in range(len(strategy)):
            for j in range(strategy[i][1]):
                w_candidates[k] =\
                    rank_tab1[strategy[i][0]][list_of_tuples[i][j]]
                k += 1

        if verbose:        
            print "-----"
            print str(w_candidates)

        ## 7.2. Solving the system
        Sol = solve_system(JF, JG, ref_vec[:s], w_candidates, verbose)
        if verbose:
            ranks.append((Sol[2], Sol[3]))
            
        if Sol[0] == None and Sol[1] == None:
            continue

        if dimension(Sol[0]) > limit:
            rejection_counter += 1
            if verbose:
                print "Kernel of dimension " + str(dimension(Sol[0])) +\
                    ". Excluded."
            continue    
        
        x0 = vector(Sol[1])
            
        if verbose:
            print "Get in the kernel of dimension : " + str(dimension(Sol[0]))

        for x in Sol[0]:
            AB =  AB_from_vector(x0 + x, m, n)
            if AB[0].is_invertible():
                A_cand = AB[0]**(-1)
                B_cand = AB[1]
                if check_candidate(JF, JG, A_cand, B_cand):
                    abC = full_tuple(F, G, A_cand, B_cand)
                    a_cand = abC[0]
                    b_cand = abC[1]
                    C_cand = abC[2]
                    G1 = EAE_function(F, A_cand, a_cand, B_cand, b_cand, C_cand)
                    if G1 == G: 
                        print "\n---------\n\nEquivalence found after " +\
                            str(counter) + " tries!\n\n------------\n"
                        return (A_cand, a_cand, B_cand, b_cand,\
                                C_cand, counter, ranks)

    print "---------\n\nNO EQUIVALENCE FOUND!\n\n----------"
    if verbose:
        print "CAUTION : more than " + str(rejection_counter) + " guesses "+\
            "led to systems with a solution space whose"
        print "dimension exceeds " + str(limit) + ", we suggest"+\
            "to try again with a higher value for"
        print "the number s of guesses\n----------\n"
    return (None, None, None, None, None, counter, ranks)
