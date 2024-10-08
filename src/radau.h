/* C-Interface to the RADAU/RADAU5-Code written by E.Hairer and G. Wanner.
   by Michael Hauth, 2001.
   mailto: Michael.Hauth@wsi-gris.uni-tuebingen.de */

#ifndef RADAU_H
#define RADAU_H
#define CPP
#ifdef CPP
extern "C" {
#endif

// #define INTEL_FORTRAN
/*#define IRIX*/    /* MipsPro Compiler */
/*#define SOLARIS*/ /* Sun Workshop Compiler */
#define GCC

/* without underscore, upper case*/
#ifdef INTEL_FORTRAN
#define UPPERCASE
#define FORTRAN_NAME(x) x
#endif

/* with underscore, lower case*/
#ifdef IRIX
#define FORTRAN_NAME(x) x##_
#endif

/* with underscore, lower case*/
#ifdef SOLARIS
#define FORTRAN_NAME(x) x##_
#endif

/* with underscore, lower case*/
#ifdef GCC
#define FORTRAN_NAME(x) x##_
#endif

#ifdef UPPERCASE
#define RADAU FORTRAN_NAME(RADAU)
#define RADAU5 FORTRAN_NAME(RADAU5)
#define CONTRA FORTRAN_NAME(CONTRA)
#define CONTR5 FORTRAN_NAME(CONTR5)
#else
#define RADAU FORTRAN_NAME(radau)
#define RADAU5 FORTRAN_NAME(radau5)
#define CONTRA FORTRAN_NAME(contra)
#define CONTR5 FORTRAN_NAME(contr5)
#endif

/* This is the direct interface to FORTRAN                     */
/*----------------------------------------------------------
 *    NUMERICAL SOLUTION OF A STIFF (OR DIFFERENTIAL ALGEBRAIC)
 *    SYSTEM OF FIRST 0RDER ORDINARY DIFFERENTIAL EQUATIONS
 *                    M*Y'=F(X,Y).
 *    THE SYSTEM CAN BE (LINEARLY) IMPLICIT (MASS-MATRIX M .NE. I)
 *    OR EXPLICIT (M=I).
 *    THE CODE IS BASED ON IMPLICIT RUNGE-KUTTA METHODS (RADAU IIA)
 *    WITH VARIABLE ORDER (5, 9, 13), WITH STEP SIZE CONTROL
 *    AND CONTINUOUS OUTPUT.
 *
 *    AUTHORS: E. HAIRER AND G. WANNER
 *             UNIVERSITE DE GENEVE, DEPT. DE MATHEMATIQUES
 *             CH-1211 GENEVE 24, SWITZERLAND
 *             E-MAIL:  Ernst.Hairer@math.unige.ch
 *                      Gerhard.Wanner@math.unige.ch
 *
 *    FOR A DESCRIPTION OF THE RELATED CODE RADAU5 SEE THE BOOK:
 *        E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
 *        EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
 *        SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS 14,
 *        SPRINGER-VERLAG 1991, SECOND EDITION 1996.
 *
 *    PRELIMINARY VERSION OF APRIL 23, 1998
 *    (latest small correction: May 6, 1999)
 *
 *    INPUT PARAMETERS
 *    ----------------
 *    N           DIMENSION OF THE SYSTEM
 *
 *    FCN         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE
 *                VALUE OF F(X,Y):
 *                   SUBROUTINE FCN(N,X,Y,F,RPAR,IPAR)
 *                   DOUBLE PRECISION X,Y(N),F(N)
 *                   F(1)=...   ETC.
 *                RPAR, IPAR (SEE BELOW)
 *
 *    X           INITIAL X-VALUE
 *
 *    Y(N)        INITIAL VALUES FOR Y
 *
 *    XEND        FINAL X-VALUE (XEND-X MAY BE POSITIVE OR NEGATIVE)
 *
 *    H           INITIAL STEP SIZE GUESS;
 *                FOR STIFF EQUATIONS WITH INITIAL TRANSIENT,
 *                H=1.D0/(NORM OF F'), USUALLY 1.D-3 OR 1.D-5, IS GOOD.
 *                THIS CHOICE IS NOT VERY IMPORTANT, THE STEP SIZE IS
 *                QUICKLY ADAPTED. (IF H=0.D0, THE CODE PUTS H=1.D-6).
 *
 *    RTOL,ATOL   RELATIVE AND ABSOLUTE ERROR TOLERANCES. THEY
 *                CAN BE BOTH SCALARS OR ELSE BOTH VECTORS OF LENGTH N.
 *
 *    ITOL        SWITCH FOR RTOL AND ATOL:
 *                  ITOL=0: BOTH RTOL AND ATOL ARE SCALARS.
 *                    THE CODE KEEPS, ROUGHLY, THE LOCAL ERROR OF
 *                    Y(I) BELOW RTOL*ABS(Y(I))+ATOL
 *                  ITOL=1: BOTH RTOL AND ATOL ARE VECTORS.
 *                    THE CODE KEEPS THE LOCAL ERROR OF Y(I) BELOW
 *                    RTOL(I)*ABS(Y(I))+ATOL(I).
 *
 *    JAC        NAME (EXTERNAL) OF THE SUBROUTINE WHICH COMPUTES
 *                THE PARTIAL DERIVATIVES OF F(X,Y) WITH RESPECT TO Y
 *                (THIS ROUTINE IS ONLY CALLED IF IJAC=1; SUPPLY
 *                A DUMMY SUBROUTINE IN THE CASE IJAC=0).
 *                FOR IJAC=1, THIS SUBROUTINE MUST HAVE THE FORM
 *                   SUBROUTINE JAC(N,X,Y,DFY,LDFY,RPAR,IPAR)
 *                   DOUBLE PRECISION X,Y(N),DFY(LDFY,N)
 *                   DFY(1,1)= ...
 *                LDFY, THE COLUMN-LENGTH OF THE ARRAY, IS
 *                FURNISHED BY THE CALLING PROGRAM.
 *                IF (MLJAC.EQ.N) THE JACOBIAN IS SUPPOSED TO
 *                   BE FULL AND THE PARTIAL DERIVATIVES ARE
 *                   STORED IN DFY AS
 *                      DFY(I,J) = PARTIAL F(I) / PARTIAL Y(J)
 *                ELSE, THE JACOBIAN IS TAKEN AS BANDED AND
 *                   THE PARTIAL DERIVATIVES ARE STORED
 *                   DIAGONAL-WISE AS
 *                      DFY(I-J+MUJAC+1,J) = PARTIAL F(I) / PARTIAL Y(J).
 *
 *    IJAC       SWITCH FOR THE COMPUTATION OF THE JACOBIAN:
 *                   IJAC=0: JACOBIAN IS COMPUTED INTERNALLY BY FINITE
 *                      DIFFERENCES, SUBROUTINE "JAC" IS NEVER CALLED.
 *                   IJAC=1: JACOBIAN IS SUPPLIED BY SUBROUTINE JAC.
 *
 *    MLJAC      SWITCH FOR THE BANDED STRUCTURE OF THE JACOBIAN:
 *                   MLJAC=N: JACOBIAN IS A FULL MATRIX. THE LINEAR
 *                      ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
 *                   0<=MLJAC<N: MLJA*IS THE LOWER BANDWITH OF JACOBIAN
 *                      MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
 *                      THE MAIN DIAGONAL).
 *
 *    MUJAC      UPPER BANDWITH OF JACOBIAN  MATRIX (>= NUMBER OF NON-
 *                ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
 *                NEED NOT BE DEFINED IF MLJAC=N.
 *
 *    ----   MAS,IMAS,MLMAS, AND MUMAS HAVE ANALOG MEANINGS      -----
 *    ----   FOR THE "MASS MATRIX" (THE MATRIX "M" OF SECTION IV.8): -
 *
 *    MAS         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE MASS-
 *                MATRIX M.
 *                IF IMAS=0, THIS MATRIX IS ASSUMED TO BE THE IDENTITY
 *                MATRIX AND NEEDS NOT TO BE DEFINED;
 *                SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
 *                IF IMAS=1, THE SUBROUTINE MAS IS OF THE FORM
 *                   SUBROUTINE MAS(N,AM,LMAS,RPAR,IPAR)
 *                   DOUBLE PRECISION AM(LMAS,N)
 *                   AM(1,1)= ....
 *                   IF (MLMAS.EQ.N) THE MASS-MATRIX IS STORED
 *                   AS FULL MATRIX LIKE
 *                        AM(I,J) = M(I,J)
 *                   ELSE, THE MATRIX IS TAKEN AS BANDED AND STORED
 *                   DIAGONAL-WISE AS
 *                        AM(I-J+MUMAS+1,J) = M(I,J).
 *
 *    IMAS       GIVES INFORMATION ON THE MASS-MATRIX:
 *                   IMAS=0: M IS SUPPOSED TO BE THE IDENTITY
 *                      MATRIX, MAS IS NEVER CALLED.
 *                   IMAS=1: MASS-MATRIX  IS SUPPLIED.
 *
 *    MLMAS       SWITCH FOR THE BANDED STRUCTURE OF THE MASS-MATRIX:
 *                   MLMAS=N: THE FULL MATRIX CASE. THE LINEAR
 *                      ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
 *                   0<=MLMAS<N: MLMAS IS THE LOWER BANDWITH OF THE
 *                      MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
 *                      THE MAIN DIAGONAL).
 *                MLMAS IS SUPPOSED TO BE .LE. MLJAC.
 *
 *    MUMAS       UPPER BANDWITH OF MASS-MATRIX (>= NUMBER OF NON-
 *                ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
 *                NEED NOT BE DEFINED IF MLMAS=N.
 *                MUMAS IS SUPPOSED TO BE .LE. MUJAC.
 *
 *    SOLOUT      NAME (EXTERNAL) OF SUBROUTINE PROVIDING THE
 *                NUMERICAL SOLUTION DURING INTEGRATION.
 *                IF IOUT=1, IT IS CALLED AFTER EVERY SUCCESSFUL STEP.
 *                SUPPLY A DUMMY SUBROUTINE IF IOUT=0.
 *                IT MUST HAVE THE FORM
 *                   SUBROUTINE SOLOUT (NR,XOLD,X,Y,CONT,LRC,N,
 *                                      RPAR,IPAR,IRTRN)
 *                   DOUBLE PRECISION X,Y(N),CONT(LRC)
 *                   ....
 *                SOLOUT FURNISHES THE SOLUTION "Y" AT THE NR-TH
 *                   GRID-POINT "X" (THEREBY THE INITIAL VALUE IS
 *                   THE FIRST GRID-POINT).
 *                "XOLD" IS THE PRECEEDING GRID-POINT.
 *                "IRTRN" SERVES TO INTERRUPT THE INTEGRATION. IF IRTRN
 *                   IS SET <0, RADAU RETURNS TO THE CALLING PROGRAM.
 *
 *         -----  CONTINUOUS OUTPUT: -----
 *                DURING CALLS TO "SOLOUT", A CONTINUOUS SOLUTION
 *                FOR THE INTERVAL [XOLD,X] IS AVAILABLE THROUGH
 *                THE FUNCTION
 *                       >>>   CONTRA(I,S,CONT,LRC)   <<<
 *                WHICH PROVIDES AN APPROXIMATION TO THE I-TH
 *                COMPONENT OF THE SOLUTION AT THE POINT S. THE VALUE
 *                S SHOULD LIE IN THE INTERVAL [XOLD,X].
 *
 *    IOUT        SWITCH FOR CALLING THE SUBROUTINE SOLOUT:
 *                   IOUT=0: SUBROUTINE IS NEVER CALLED
 *                   IOUT=1: SUBROUTINE IS AVAILABLE FOR OUTPUT.
 *
 *    WORK        ARRAY OF WORKING SPACE OF LENGTH "LWORK".
 *                WORK(1), WORK(2),.., WORK(20) SERVE AS PARAMETERS
 *                FOR THE CODE. FOR STANDARD USE OF THE CODE
 *                WORK(1),..,WORK(20) MUST BE SET TO ZERO BEFORE THE
 *                FIRST CALL. SEE BELOW FOR A MORE SOPHISTICATED USE.
 *                WORK(8),..,WORK(LWORK) SERVE AS WORKING SPACE
 *                FOR ALL VECTORS AND MATRICES.
 *                "LWORK" MUST BE AT LEAST
 *                         N*(LJAC+LMAS+NSMAX*LE+3*NSMAX+3)+20
 *                WHERE
 *                   NSMAX=IWORK(12) (SEE BELOW)
 *                AND
 *                   LJAC=N              IF MLJAC=N (FULL JACOBIAN)
 *                   LJAC=MLJAC+MUJAC+1  IF MLJAC<N (BANDED JAC.)
 *                AND
 *                   LMAS=0              IF IMAS=0
 *                   LMAS=N              IF IMAS=1 AND MLMAS=N (FULL)
 *                   LMAS=MLMAS+MUMAS+1  IF MLMAS<N (BANDED MASS-M.)
 *                AND
 *                   LE=N               IF MLJAC=N (FULL JACOBIAN)
 *                   LE=2*MLJAC+MUJAC+1 IF MLJAC<N (BANDED JAC.)
 *
 *                IN THE USUAL CASE WHERE THE JACOBIAN IS FULL AND THE
 *                MASS-MATRIX IS THE INDENTITY (IMAS=0), THE MINIMUM
 *                STORAGE REQUIREMENT IS
 *                     LWORK = (NSMAX+1)*N*N+(3*NSMAX+3)*N+20.
 *                IF IWORK(9)=M1>0 THEN "LWORK" MUST BE AT LEAST
 *                     N*(LJAC+3*NSMAX+3)+(N-M1)*(LMAS+NSMAX*LE)+20
 *                WHERE IN THE DEFINITIONS OF LJAC, LMAS AND LE THE
 *                NUMBER N CAN BE REPLACED BY N-M1.
 *
 *    LWORK       DECLARED LENGTH OF ARRAY "WORK".
 *
 *    IWORK       INTEGER WORKING SPACE OF LENGTH "LIWORK".
 *                IWORK(1),IWORK(2),...,IWORK(20) SERVE AS PARAMETERS
 *                FOR THE CODE. FOR STANDARD USE, SET IWORK(1),..,
 *                IWORK(20) TO ZERO BEFORE CALLING.
 *                IWORK(21),...,IWORK(LIWORK) SERVE AS WORKING AREA.
 *                "LIWORK" MUST BE AT LEAST
 *                            (2+(NSMAX-1)/2)*N+20.
 *
 *    LIWORK      DECLARED LENGTH OF ARRAY "IWORK".
 *
 *    RPAR, IPAR  REAL AND INTEGER PARAMETERS (OR PARAMETER ARRAYS) WHICH
 *                CAN BE USED FOR COMMUNICATION BETWEEN YOUR CALLING
 *                PROGRAM AND THE FCN, JAC, MAS, SOLOUT SUBROUTINES.
 *
 *----------------------------------------------------------------------
 *
 *    SOPHISTICATED SETTING OF PARAMETERS
 *    -----------------------------------
 *             SEVERAL PARAMETERS OF THE CODE ARE TUNED TO MAKE IT WORK
 *             WELL. THEY MAY BE DEFINED BY SETTING WORK(1),...
 *             AS WELL AS IWORK(1),... DIFFERENT FROM ZERO.
 *             FOR ZERO INPUT, THE CODE CHOOSES DEFAULT VALUES:
 *
 *   IWORK(1)  IF IWORK(1).NE.0, THE CODE TRANSFORMS THE JACOBIAN
 *             MATRIX TO HESSENBERG FORM. THIS IS PARTICULARLY
 *             ADVANTAGEOUS FOR LARGE SYSTEMS WITH FULL JACOBIAN.
 *             IT DOES NOT WORK FOR BANDED JACOBIAN (MLJAC<N)
 *             AND NOT FOR IMPLICIT SYSTEMS (IMAS=1).
 *
 *   IWORK(2)  THIS IS THE MAXIMAL NUMBER OF ALLOWED STEPS.
 *             THE DEFAULT VALUE (FOR IWORK(2)=0) IS 100000.
 *
 *   IWORK(3)  THE MAXIMUM NUMBER OF NEWTON ITERATIONS FOR THE
 *             SOLUTION OF THE IMPLICIT SYSTEM IN EACH STEP
 *             IWORK(3)+(NS-3)*2.5. DEFAULT VALUE (FOR IWORK(3)=0) IS 7.
 *             NS IS THE NUMBER OF STAGES (SEE IWORK(11)).
 *
 *   IWORK(4)  IF IWORK(4).EQ.0 THE EXTRAPOLATED COLLOCATION SOLUTION
 *             IS TAKEN AS STARTING VALUE FOR NEWTON'S METHOD.
 *             IF IWORK(4).NE.0 ZERO STARTING VALUES ARE USED.
 *             THE LATTER IS RECOMMENDED IF NEWTON'S METHOD HAS
 *             DIFFICULTIES WITH CONVERGENCE (THIS IS THE CASE WHEN
 *             NSTEP IS LARGER THAN NACCPT + NREJCT; SEE OUTPUT PARAM.).
 *             DEFAULT IS IWORK(4)=0.
 *
 *      THE FOLLOWING 3 PARAMETERS ARE IMPORTANT FOR
 *      DIFFERENTIAL-ALGEBRAI*SYSTEMS OF INDEX > 1.
 *      THE FUNCTION-SUBROUTINE SHOULD BE WRITTEN SUCH THAT
 *      THE INDEX 1,2,3 VARIABLES APPEAR IN THIS ORDER.
 *      IN ESTIMATING THE ERROR THE INDEX 2 VARIABLES ARE
 *      MULTIPLIED BY H, THE INDEX 3 VARIABLES BY H**2.
 *
 *   IWORK(5)  DIMENSION OF THE INDEX 1 VARIABLES (MUST BE > 0). FOR
 *             ODE'S THIS EQUALS THE DIMENSION OF THE SYSTEM.
 *             DEFAULT IWORK(5)=N.
 *
 *   IWORK(6)  DIMENSION OF THE INDEX 2 VARIABLES. DEFAULT IWORK(6)=0.
 *
 *   IWORK(7)  DIMENSION OF THE INDEX 3 VARIABLES. DEFAULT IWORK(7)=0.
 *
 *   IWORK(8)  SWITCH FOR STEP SIZE STRATEGY
 *             IF IWORK(8).EQ.1  MOD. PREDICTIVE CONTROLLER (GUSTAFSSON)
 *             IF IWORK(8).EQ.2  CLASSICAL STEP SIZE CONTROL
 *             THE DEFAULT VALUE (FOR IWORK(8)=0) IS IWORK(8)=1.
 *             THE CHOICE IWORK(8).EQ.1 SEEMS TO PRODUCE SAFER RESULTS;
 *             FOR SIMPLE PROBLEMS, THE CHOICE IWORK(8).EQ.2 PRODUCES
 *             OFTEN SLIGHTLY FASTER RUNS
 *
 *      IF THE DIFFERENTIAL SYSTEM HAS THE SPECIAL STRUCTURE THAT
 *           Y(I)' = Y(I+M2)   FOR  I=1,...,M1,
 *      WITH M1 A MULTIPLE OF M2, A SUBSTANTIAL GAIN IN COMPUTERTIME
 *      CAN BE ACHIEVED BY SETTING THE PARAMETERS IWORK(9) AND IWORK(10).
 *      E.G., FOR SECOND ORDER SYSTEMS P'=V, V'=G(P,V), WHERE P AND V ARE
 *      VECTORS OF DIMENSION N/2, ONE HAS TO PUT M1=M2=N/2.
 *      FOR M1>0 SOME OF THE INPUT PARAMETERS HAVE DIFFERENT MEANINGS:
 *      - JAC: ONLY THE ELEMENTS OF THE NON-TRIVIAL PART OF THE
 *             JACOBIAN HAVE TO BE STORED
 *             IF (MLJAC.EQ.N-M1) THE JACOBIAN IS SUPPOSED TO BE FULL
 *                DFY(I,J) = PARTIAL F(I+M1) / PARTIAL Y(J)
 *               FOR I=1,N-M1 AND J=1,N.
 *             ELSE, THE JACOBIAN IS BANDED ( M1 = M2 * MM )
 *                DFY(I-J+MUJAC+1,J+K*M2) = PARTIAL F(I+M1) / PARTIAL Y(J+K*M2)
 *               FOR I=1,MLJAC+MUJAC+1 AND J=1,M2 AND K=0,MM.
 *      - MLJAC: MLJAC=N-M1: IF THE NON-TRIVIAL PART OF THE JACOBIAN IS FULL
 *               0<=MLJAC<N-M1: IF THE (MM+1) SUBMATRICES (FOR K=0,MM)
 *                    PARTIAL F(I+M1) / PARTIAL Y(J+K*M2),  I,J=1,M2
 *                   ARE BANDED, MLJA*IS THE MAXIMAL LOWER BANDWIDTH
 *                   OF THESE MM+1 SUBMATRICES
 *      - MUJAC: MAXIMAL UPPER BANDWIDTH OF THESE MM+1 SUBMATRICES
 *               NEED NOT BE DEFINED IF MLJAC=N-M1
 *      - MAS: IF IMAS=0 THIS MATRIX IS ASSUMED TO BE THE IDENTITY AND
 *             NEED NOT BE DEFINED. SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
 *             IT IS ASSUMED THAT ONLY THE ELEMENTS OF RIGHT LOWER BLOCK OF
 *             DIMENSION N-M1 DIFFER FROM THAT OF THE IDENTITY MATRIX.
 *             IF (MLMAS.EQ.N-M1) THIS SUBMATRIX IS SUPPOSED TO BE FULL
 *                AM(I,J) = M(I+M1,J+M1)     FOR I=1,N-M1 AND J=1,N-M1.
 *             ELSE, THE MASS MATRIX IS BANDED
 *                AM(I-J+MUMAS+1,J) = M(I+M1,J+M1)
 *      - MLMAS: MLMAS=N-M1: IF THE NON-TRIVIAL PART OF M IS FULL
 *               0<=MLMAS<N-M1: LOWER BANDWIDTH OF THE MASS MATRIX
 *      - MUMAS: UPPER BANDWIDTH OF THE MASS MATRIX
 *               NEED NOT BE DEFINED IF MLMAS=N-M1
 *
 *   IWORK(9)  THE VALUE OF M1.  DEFAULT M1=0.
 *
 *   IWORK(10) THE VALUE OF M2.  DEFAULT M2=M1.
 *
 *   IWORK(11) NSMIN, MINIMAL NUMBER OF STAGES NS (ORDER 2*NS-1)
 *             POSSIBLE VALUES ARE 1,3,5,7. DEFAULT NS=3.
 *
 *   IWORK(12) NSMAX, MAXIMAL NUMBER OF STAGES NS.
 *             POSSIBLE VALUES ARE 1,3,5,7. DEFAULT NS=7.
 *
 *   IWORK(13) VALUE OF NS FOR THE FIRST STEP (DEFAULT VALUE: NSMIN)
 *
 *----------
 *
 *   WORK(1)   UROUND, THE ROUNDING UNIT, DEFAULT 1.D-16.
 *
 *   WORK(2)   THE SAFETY FACTOR IN STEP SIZE PREDICTION,
 *             DEFAULT 0.9D0.
 *
 *   WORK(3)   DECIDES WHETHER THE JACOBIAN SHOULD BE RECOMPUTED;
 *             INCREASE WORK(3), TO 0.1 SAY, WHEN JACOBIAN EVALUATIONS
 *             ARE COSTLY. FOR SMALL SYSTEMS WORK(3) SHOULD BE SMALLER
 *             (0.001D0, SAY). NEGATIV WORK(3) FORCES THE CODE TO
 *             COMPUTE THE JACOBIAN AFTER EVERY ACCEPTED STEP.
 *             DEFAULT 0.001D0.
 *
 *   WORK(5) AND WORK(6) : IF WORK(5) < HNEW/HOLD < WORK(6), THEN THE
 *             STEP SIZE IS NOT CHANGED. THIS SAVES, TOGETHER WITH A
 *             LARGE WORK(3), LU-DECOMPOSITIONS AND COMPUTING TIME FOR
 *             LARGE SYSTEMS. FOR SMALL SYSTEMS ONE MAY HAVE
 *             WORK(5)=1.D0, WORK(6)=1.2D0, FOR LARGE FULL SYSTEMS
 *             WORK(5)=0.99D0, WORK(6)=2.D0 MIGHT BE GOOD.
 *             DEFAULTS WORK(5)=1.D0, WORK(6)=1.2D0 .
 *
 *   WORK(7)   MAXIMAL STEP SIZE, DEFAULT XEND-X.
 *
 *   WORK(8), WORK(9)   PARAMETERS FOR STEP SIZE SELECTION
 *             THE NEW STEP SIZE IS CHOSEN SUBJECT TO THE RESTRICTION
 *                WORK(8) <= HNEW/HOLD <= WORK(9)
 *             DEFAULT VALUES: WORK(8)=0.2D0, WORK(9)=8.D0
 *
 *   WORK(10)  ORDER IS INCREASED IF THE CONTRACTIVITY FACTOR IS
 *             SMALL THAN WORK(10), DEFAULT VALUE IS 0.002
 *
 *   WORK(11)  ORDER IS DECREASED IF THE CONTRACTIVITY FACTOR IS
 *             LARGER THAN WORK(11), DEFAULT VALUE IS 0.8
 *
 *   WORK(12), WORK(13)  ORDER IS DECREASED ONLY IF THE STEPSIZE
 *             RATIO SATISFIES  WORK(13).LE.HNEW/H.LE.WORK(12),
 *             DEFAULT VALUES ARE 1.2 AND 0.8
 *
 *-----------------------------------------------------------------------
 *
 *    OUTPUT PARAMETERS
 *    -----------------
 *    X           X-VALUE FOR WHICH THE SOLUTION HAS BEEN COMPUTED
 *                (AFTER SUCCESSFUL RETURN X=XEND).
 *
 *    Y(N)        NUMERICAL SOLUTION AT X
 *
 *    H           PREDICTED STEP SIZE OF THE LAST ACCEPTED STEP
 *
 *    IDID        REPORTS ON SUCCESSFULNESS UPON RETURN:
 *                  IDID= 1  COMPUTATION SUCCESSFUL,
 *                  IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)
 *                  IDID=-1  INPUT IS NOT CONSISTENT,
 *                  IDID=-2  LARGER NMAX IS NEEDED,
 *                  IDID=-3  STEP SIZE BECOMES TOO SMALL,
 *                  IDID=-4  MATRIX IS REPEATEDLY SINGULAR.
 *
 *  IWORK(14)  NFCN    NUMBER OF FUNCTION EVALUATIONS (THOSE FOR NUMERICAL
 *                     EVALUATION OF THE JACOBIAN ARE NOT COUNTED)
 *  IWORK(15)  NJAC    NUMBER OF JACOBIAN EVALUATIONS (EITHER ANALYTICALLY
 *                     OR NUMERICALLY)
 *  IWORK(16)  NSTEP   NUMBER OF COMPUTED STEPS
 *  IWORK(17)  NACCPT  NUMBER OF ACCEPTED STEPS
 *  IWORK(18)  NREJCT  NUMBER OF REJECTED STEPS (DUE TO ERROR TEST),
 *                     (STEP REJECTIONS IN THE FIRST STEP ARE NOT COUNTED)
 *  IWORK(19)  NDEC    NUMBER OF LU-DECOMPOSITIONS OF THE MATRICES
 *  IWORK(20)  NSOL    NUMBER OF FORWARD-BACKWARD SUBSTITUTIONS, OF ALL
 *                     SYSTEMS THAT HAVE TO BE SOLVED FOR ONE SIMPLIFIED
 *                     NEWTON ITERATION; THE NSTEP (REAL) FORWARD-BACKWARD
 *                     SUBSTITUTIONS, NEEDED FOR STEP SIZE SELECTION,
 *                     ARE NOT COUNTED.
 *-----------------------------------------------------------------------*/

void RADAU(int* N,
           void FCN(int*, double*, double*, double*, double*, int*),
           double* X, double* Y, double* XEND, double* H,
           double* RTOL, double* ATOL, int* ITOL,
           void JAC(int*, double*, double*, double*, int*, double*, double*),
           int* IJAC, int* MLJAC, int* MUJAC,
           void MAS(int* n, double* am, int* lmas, int* rpar, int* ipar),
           int* IMAS, int* MLMAS, int* MUMAS,
           void SOLOUT(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*),
           int* IOUT,
           double* WORK, int* LWORK, int* IWORK, int* LIWORK,
           double* RPAR, int* IPAR, int* IDID);
/* C-Interface to radau.
   Cares for memory allocation and parameter passing.
   It is possible to supply constants as parameters.
   Work and Iwork must point to arrays of at least 20 elements.
   (for Input/Output parameters only)
*/
void cradau(int n,
            void fcn(int*, double*, double*, double*, double*, int*),
            double x, double* y, double xend, double h,
            double rtol, double atol,
            void jac(int*, double*, double*, double*, int*, double*, double*),
            int ijac, int mljac, int mujac,
            void mas(int* n, double* am, int* lmas, int* rpar, int* ipar),
            int imas, int mlmas, int mumas,
            void solout(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*),
            int iout,
            double* work, int* iwork,
            double* rpar, int* ipar, int* idid);

/* Function for testing purposes.
   This (FORTRAN) function possesses the same interface as RADAU
   and just prints out the parameters. */
void RADAU_TEST(int* N,
                void FCN(int*, double*, double*, double*, double*, int*),
                double* X, double* Y, double* XEND, double* H,
                double* RTOL, double* ATOL, int* ITOL,
                void JAC(int*, double*, double*, double*, int*, double*, double*),
                int* IJAC, int* MLJAC, int* MUJAC,
                void MAS(int* n, double* am, int* lmas, int* rpar, int* ipar),
                int* IMAS, int* MLMAS, int* MUMAS,
                void SOLOUT(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*),
                int* IOUT,
                double* WORK, int* LWORK, int* IWORK, int* LIWORK,
                double* RPAR, int* IPAR, int* IDID);

/* Interface to the FORTRAN function for contignuous output.(see above) */
double CONTRA(int* I, double* S, double* CONT, int* LRC);
/* C-Interface to CONTRA.
   Cares for parameter passing */
double ccontra(int i, double s, double* cont, int* lrc);

/* ----------------------------------------------------------
 *     NUMERICAL SOLUTION OF A STIFF (OR DIFFERENTIAL ALGEBRAIC)
 *     SYSTEM OF FIRST 0RDER ORDINARY DIFFERENTIAL EQUATIONS
 *                     M*Y'=F(X,Y).
 *     THE SYSTEM CAN BE (LINEARLY) IMPLICIT (MASS-MATRIX M .NE. I)
 *     OR EXPLICIT (M=I).
 *     THE METHOD USED IS AN IMPLICIT RUNGE-KUTTA METHOD (RADAU IIA)
 *     OF ORDER 5 WITH STEP SIZE CONTROL AND CONTINUOUS OUTPUT.
 *     CF. SECTION IV.8
 *
 *     AUTHORS: E. HAIRER AND G. WANNER
 *              UNIVERSITE DE GENEVE, DEPT. DE MATHEMATIQUES
 *              CH-1211 GENEVE 24, SWITZERLAND
 *              E-MAIL:  Ernst.Hairer@math.unige.ch
 *                       Gerhard.Wanner@math.unige.ch
 *
 *     THIS CODE IS PART OF THE BOOK:
 *         E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
 *         EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
 *         SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS 14,
 *         SPRINGER-VERLAG 1991, SECOND EDITION 1996.
 *
 *     VERSION OF JULY 9, 1996
 *        (small correction April 14, 2000)
 *
 *     INPUT PARAMETERS
 *     ----------------
 *     N           DIMENSION OF THE SYSTEM
 *
 *     FCN         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE
 *                 VALUE OF F(X,Y):
 *                    SUBROUTINE FCN(N,X,Y,F,RPAR,IPAR)
 *                    DOUBLE PRECISION X,Y(N),F(N)
 *                    F(1)=...   ETC.
 *                 RPAR, IPAR (SEE BELOW)
 *
 *     X           INITIAL X-VALUE
 *
 *     Y(N)        INITIAL VALUES FOR Y
 *
 *     XEND        FINAL X-VALUE (XEND-X MAY BE POSITIVE OR NEGATIVE)
 *
 *     H           INITIAL STEP SIZE GUESS;
 *                 FOR STIFF EQUATIONS WITH INITIAL TRANSIENT,
 *                 H=1.D0/(NORM OF F'), USUALLY 1.D-3 OR 1.D-5, IS GOOD.
 *                 THIS CHOICE IS NOT VERY IMPORTANT, THE STEP SIZE IS
 *                 QUICKLY ADAPTED. (IF H=0.D0, THE CODE PUTS H=1.D-6).
 *
 *     RTOL,ATOL   RELATIVE AND ABSOLUTE ERROR TOLERANCES. THEY
 *                 CAN BE BOTH SCALARS OR ELSE BOTH VECTORS OF LENGTH N.
 *
 *     ITOL        SWITCH FOR RTOL AND ATOL:
 *                   ITOL=0: BOTH RTOL AND ATOL ARE SCALARS.
 *                     THE CODE KEEPS, ROUGHLY, THE LOCAL ERROR OF
 *                     Y(I) BELOW RTOL*ABS(Y(I))+ATOL
 *                   ITOL=1: BOTH RTOL AND ATOL ARE VECTORS.
 *                     THE CODE KEEPS THE LOCAL ERROR OF Y(I) BELOW
 *                     RTOL(I)*ABS(Y(I))+ATOL(I).
 *
 *     JAC         NAME (EXTERNAL) OF THE SUBROUTINE WHICH COMPUTES
 *                 THE PARTIAL DERIVATIVES OF F(X,Y) WITH RESPECT TO Y
 *                 (THIS ROUTINE IS ONLY CALLED IF IJAC=1; SUPPLY
 *                 A DUMMY SUBROUTINE IN THE CASE IJAC=0).
 *                 FOR IJAC=1, THIS SUBROUTINE MUST HAVE THE FORM
 *                    SUBROUTINE JAC(N,X,Y,DFY,LDFY,RPAR,IPAR)
 *                    DOUBLE PRECISION X,Y(N),DFY(LDFY,N)
 *                    DFY(1,1)= ...
 *                 LDFY, THE COLUMN-LENGTH OF THE ARRAY, IS
 *                 FURNISHED BY THE CALLING PROGRAM.
 *                 IF (MLJAC.EQ.N) THE JACOBIAN IS SUPPOSED TO
 *                    BE FULL AND THE PARTIAL DERIVATIVES ARE
 *                    STORED IN DFY AS
 *                       DFY(I,J) = PARTIAL F(I) / PARTIAL Y(J)
 *                 ELSE, THE JACOBIAN IS TAKEN AS BANDED AND
 *                    THE PARTIAL DERIVATIVES ARE STORED
 *                    DIAGONAL-WISE AS
 *                       DFY(I-J+MUJAC+1,J) = PARTIAL F(I) / PARTIAL Y(J).
 *
 *     IJAC        SWITCH FOR THE COMPUTATION OF THE JACOBIAN:
 *                    IJAC=0: JACOBIAN IS COMPUTED INTERNALLY BY FINITE
 *                       DIFFERENCES, SUBROUTINE "JAC" IS NEVER CALLED.
 *                    IJAC=1: JACOBIAN IS SUPPLIED BY SUBROUTINE JAC.
 *
 *     MLJAC       SWITCH FOR THE BANDED STRUCTURE OF THE JACOBIAN:
 *                    MLJAC=N: JACOBIAN IS A FULL MATRIX. THE LINEAR
 *                       ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
 *                    0<=MLJAC<N: MLJAC IS THE LOWER BANDWITH OF JACOBIAN
 *                       MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
 *                       THE MAIN DIAGONAL).
 *
 *     MUJAC       UPPER BANDWITH OF JACOBIAN  MATRIX (>= NUMBER OF NON-
 *                 ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
 *                 NEED NOT BE DEFINED IF MLJAC=N.
 *
 *     ----   MAS,IMAS,MLMAS, AND MUMAS HAVE ANALOG MEANINGS      -----
 *     ----   FOR THE "MASS MATRIX" (THE MATRIX "M" OF SECTION IV.8): -
 *
 *     MAS         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE MASS-
 *                 MATRIX M.
 *                 IF IMAS=0, THIS MATRIX IS ASSUMED TO BE THE IDENTITY
 *                 MATRIX AND NEEDS NOT TO BE DEFINED;
 *                 SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
 *                 IF IMAS=1, THE SUBROUTINE MAS IS OF THE FORM
 *                    SUBROUTINE MAS(N,AM,LMAS,RPAR,IPAR)
 *                    DOUBLE PRECISION AM(LMAS,N)
 *                    AM(1,1)= ....
 *                    IF (MLMAS.EQ.N) THE MASS-MATRIX IS STORED
 *                    AS FULL MATRIX LIKE
 *                         AM(I,J) = M(I,J)
 *                    ELSE, THE MATRIX IS TAKEN AS BANDED AND STORED
 *                    DIAGONAL-WISE AS
 *                         AM(I-J+MUMAS+1,J) = M(I,J).
 *
 *     IMAS       GIVES INFORMATION ON THE MASS-MATRIX:
 *                    IMAS=0: M IS SUPPOSED TO BE THE IDENTITY
 *                       MATRIX, MAS IS NEVER CALLED.
 *                    IMAS=1: MASS-MATRIX  IS SUPPLIED.
 *
 *     MLMAS       SWITCH FOR THE BANDED STRUCTURE OF THE MASS-MATRIX:
 *                    MLMAS=N: THE FULL MATRIX CASE. THE LINEAR
 *                       ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
 *                    0<=MLMAS<N: MLMAS IS THE LOWER BANDWITH OF THE
 *                       MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
 *                       THE MAIN DIAGONAL).
 *                 MLMAS IS SUPPOSED TO BE .LE. MLJAC.
 *
 *     MUMAS       UPPER BANDWITH OF MASS-MATRIX (>= NUMBER OF NON-
 *                 ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
 *                 NEED NOT BE DEFINED IF MLMAS=N.
 *                 MUMAS IS SUPPOSED TO BE .LE. MUJAC.
 *
 *     SOLOUT      NAME (EXTERNAL) OF SUBROUTINE PROVIDING THE
 *                 NUMERICAL SOLUTION DURING INTEGRATION.
 *                 IF IOUT=1, IT IS CALLED AFTER EVERY SUCCESSFUL STEP.
 *                 SUPPLY A DUMMY SUBROUTINE IF IOUT=0.
 *                 IT MUST HAVE THE FORM
 *                    SUBROUTINE SOLOUT (NR,XOLD,X,Y,CONT,LRC,N,
 *                                       RPAR,IPAR,IRTRN)
 *                    DOUBLE PRECISION X,Y(N),CONT(LRC)
 *                    ....
 *                 SOLOUT FURNISHES THE SOLUTION "Y" AT THE NR-TH
 *                    GRID-POINT "X" (THEREBY THE INITIAL VALUE IS
 *                    THE FIRST GRID-POINT).
 *                 "XOLD" IS THE PRECEEDING GRID-POINT.
 *                 "IRTRN" SERVES TO INTERRUPT THE INTEGRATION. IF IRTRN
 *                    IS SET <0, RADAU5 RETURNS TO THE CALLING PROGRAM.
 *
 *          -----  CONTINUOUS OUTPUT: -----
 *                 DURING CALLS TO "SOLOUT", A CONTINUOUS SOLUTION
 *                 FOR THE INTERVAL [XOLD,X] IS AVAILABLE THROUGH
 *                 THE FUNCTION
 *                        >>>   CONTR5(I,S,CONT,LRC)   <<<
 *                 WHICH PROVIDES AN APPROXIMATION TO THE I-TH
 *                 COMPONENT OF THE SOLUTION AT THE POINT S. THE VALUE
 *                 S SHOULD LIE IN THE INTERVAL [XOLD,X].
 *                 DO NOT CHANGE THE ENTRIES OF CONT(LRC), IF THE
 *                 DENSE OUTPUT FUNCTION IS USED.
 *
 *     IOUT        SWITCH FOR CALLING THE SUBROUTINE SOLOUT:
 *                    IOUT=0: SUBROUTINE IS NEVER CALLED
 *                    IOUT=1: SUBROUTINE IS AVAILABLE FOR OUTPUT.
 *
 *     WORK        ARRAY OF WORKING SPACE OF LENGTH "LWORK".
 *                 WORK(1), WORK(2),.., WORK(20) SERVE AS PARAMETERS
 *                 FOR THE CODE. FOR STANDARD USE OF THE CODE
 *                 WORK(1),..,WORK(20) MUST BE SET TO ZERO BEFORE
 *                 CALLING. SEE BELOW FOR A MORE SOPHISTICATED USE.
 *                 WORK(21),..,WORK(LWORK) SERVE AS WORKING SPACE
 *                 FOR ALL VECTORS AND MATRICES.
 *                 "LWORK" MUST BE AT LEAST
 *                             N*(LJAC+LMAS+3*LE+12)+20
 *                 WHERE
 *                    LJAC=N              IF MLJAC=N (FULL JACOBIAN)
 *                    LJAC=MLJAC+MUJAC+1  IF MLJAC<N (BANDED JAC.)
 *                 AND
 *                    LMAS=0              IF IMAS=0
 *                    LMAS=N              IF IMAS=1 AND MLMAS=N (FULL)
 *                    LMAS=MLMAS+MUMAS+1  IF MLMAS<N (BANDED MASS-M.)
 *                 AND
 *                    LE=N               IF MLJAC=N (FULL JACOBIAN)
 *                    LE=2*MLJAC+MUJAC+1 IF MLJAC<N (BANDED JAC.)
 *
 *                 IN THE USUAL CASE WHERE THE JACOBIAN IS FULL AND THE
 *                 MASS-MATRIX IS THE INDENTITY (IMAS=0), THE MINIMUM
 *                 STORAGE REQUIREMENT IS
 *                             LWORK = 4*N*N+12*N+20.
 *                 IF IWORK(9)=M1>0 THEN "LWORK" MUST BE AT LEAST
 *                          N*(LJAC+12)+(N-M1)*(LMAS+3*LE)+20
 *                 WHERE IN THE DEFINITIONS OF LJAC, LMAS AND LE THE
 *                 NUMBER N CAN BE REPLACED BY N-M1.
 *
 *     LWORK       DECLARED LENGTH OF ARRAY "WORK".
 *
 *     IWORK       INTEGER WORKING SPACE OF LENGTH "LIWORK".
 *                 IWORK(1),IWORK(2),...,IWORK(20) SERVE AS PARAMETERS
 *                 FOR THE CODE. FOR STANDARD USE, SET IWORK(1),..,
 *                 IWORK(20) TO ZERO BEFORE CALLING.
 *                 IWORK(21),...,IWORK(LIWORK) SERVE AS WORKING AREA.
 *                 "LIWORK" MUST BE AT LEAST 3*N+20.
 *
 *     LIWORK      DECLARED LENGTH OF ARRAY "IWORK".
 *
 *     RPAR, IPAR  REAL AND INTEGER PARAMETERS (OR PARAMETER ARRAYS) WHICH
 *                 CAN BE USED FOR COMMUNICATION BETWEEN YOUR CALLING
 *                 PROGRAM AND THE FCN, JAC, MAS, SOLOUT SUBROUTINES.
 *
 * ----------------------------------------------------------------------
 *
 *     SOPHISTICATED SETTING OF PARAMETERS
 *     -----------------------------------
 *              SEVERAL PARAMETERS OF THE CODE ARE TUNED TO MAKE IT WORK
 *              WELL. THEY MAY BE DEFINED BY SETTING WORK(1),...
 *              AS WELL AS IWORK(1),... DIFFERENT FROM ZERO.
 *              FOR ZERO INPUT, THE CODE CHOOSES DEFAULT VALUES:
 *
 *    IWORK(1)  IF IWORK(1).NE.0, THE CODE TRANSFORMS THE JACOBIAN
 *              MATRIX TO HESSENBERG FORM. THIS IS PARTICULARLY
 *              ADVANTAGEOUS FOR LARGE SYSTEMS WITH FULL JACOBIAN.
 *              IT DOES NOT WORK FOR BANDED JACOBIAN (MLJAC<N)
 *              AND NOT FOR IMPLICIT SYSTEMS (IMAS=1).
 *
 *    IWORK(2)  THIS IS THE MAXIMAL NUMBER OF ALLOWED STEPS.
 *              THE DEFAULT VALUE (FOR IWORK(2)=0) IS 100000.
 *
 *    IWORK(3)  THE MAXIMUM NUMBER OF NEWTON ITERATIONS FOR THE
 *              SOLUTION OF THE IMPLICIT SYSTEM IN EACH STEP.
 *              THE DEFAULT VALUE (FOR IWORK(3)=0) IS 7.
 *
 *    IWORK(4)  IF IWORK(4).EQ.0 THE EXTRAPOLATED COLLOCATION SOLUTION
 *              IS TAKEN AS STARTING VALUE FOR NEWTON'S METHOD.
 *              IF IWORK(4).NE.0 ZERO STARTING VALUES ARE USED.
 *              THE LATTER IS RECOMMENDED IF NEWTON'S METHOD HAS
 *              DIFFICULTIES WITH CONVERGENCE (THIS IS THE CASE WHEN
 *              NSTEP IS LARGER THAN NACCPT + NREJCT; SEE OUTPUT PARAM.).
 *              DEFAULT IS IWORK(4)=0.
 *
 *       THE FOLLOWING 3 PARAMETERS ARE IMPORTANT FOR
 *       DIFFERENTIAL-ALGEBRAIC SYSTEMS OF INDEX > 1.
 *       THE FUNCTION-SUBROUTINE SHOULD BE WRITTEN SUCH THAT
 *       THE INDEX 1,2,3 VARIABLES APPEAR IN THIS ORDER.
 *       IN ESTIMATING THE ERROR THE INDEX 2 VARIABLES ARE
 *       MULTIPLIED BY H, THE INDEX 3 VARIABLES BY H**2.
 *
 *    IWORK(5)  DIMENSION OF THE INDEX 1 VARIABLES (MUST BE > 0). FOR
 *              ODE'S THIS EQUALS THE DIMENSION OF THE SYSTEM.
 *              DEFAULT IWORK(5)=N.
 *
 *    IWORK(6)  DIMENSION OF THE INDEX 2 VARIABLES. DEFAULT IWORK(6)=0.
 *
 *    IWORK(7)  DIMENSION OF THE INDEX 3 VARIABLES. DEFAULT IWORK(7)=0.
 *
 *    IWORK(8)  SWITCH FOR STEP SIZE STRATEGY
 *              IF IWORK(8).EQ.1  MOD. PREDICTIVE CONTROLLER (GUSTAFSSON)
 *              IF IWORK(8).EQ.2  CLASSICAL STEP SIZE CONTROL
 *              THE DEFAULT VALUE (FOR IWORK(8)=0) IS IWORK(8)=1.
 *              THE CHOICE IWORK(8).EQ.1 SEEMS TO PRODUCE SAFER RESULTS;
 *              FOR SIMPLE PROBLEMS, THE CHOICE IWORK(8).EQ.2 PRODUCES
 *              OFTEN SLIGHTLY FASTER RUNS
 *
 *       IF THE DIFFERENTIAL SYSTEM HAS THE SPECIAL STRUCTURE THAT
 *            Y(I)' = Y(I+M2)   FOR  I=1,...,M1,
 *       WITH M1 A MULTIPLE OF M2, A SUBSTANTIAL GAIN IN COMPUTERTIME
 *       CAN BE ACHIEVED BY SETTING THE PARAMETERS IWORK(9) AND IWORK(10).
 *       E.G., FOR SECOND ORDER SYSTEMS P'=V, V'=G(P,V), WHERE P AND V ARE
 *       VECTORS OF DIMENSION N/2, ONE HAS TO PUT M1=M2=N/2.
 *       FOR M1>0 SOME OF THE INPUT PARAMETERS HAVE DIFFERENT MEANINGS:
 *       - JAC: ONLY THE ELEMENTS OF THE NON-TRIVIAL PART OF THE
 *              JACOBIAN HAVE TO BE STORED
 *              IF (MLJAC.EQ.N-M1) THE JACOBIAN IS SUPPOSED TO BE FULL
 *                 DFY(I,J) = PARTIAL F(I+M1) / PARTIAL Y(J)
 *                FOR I=1,N-M1 AND J=1,N.
 *              ELSE, THE JACOBIAN IS BANDED ( M1 = M2 * MM )
 *                 DFY(I-J+MUJAC+1,J+K*M2) = PARTIAL F(I+M1) / PARTIAL Y(J+K*M2)
 *                FOR I=1,MLJAC+MUJAC+1 AND J=1,M2 AND K=0,MM.
 *       - MLJAC: MLJAC=N-M1: IF THE NON-TRIVIAL PART OF THE JACOBIAN IS FULL
 *                0<=MLJAC<N-M1: IF THE (MM+1) SUBMATRICES (FOR K=0,MM)
 *                     PARTIAL F(I+M1) / PARTIAL Y(J+K*M2),  I,J=1,M2
 *                    ARE BANDED, MLJAC IS THE MAXIMAL LOWER BANDWIDTH
 *                    OF THESE MM+1 SUBMATRICES
 *       - MUJAC: MAXIMAL UPPER BANDWIDTH OF THESE MM+1 SUBMATRICES
 *                NEED NOT BE DEFINED IF MLJAC=N-M1
 *       - MAS: IF IMAS=0 THIS MATRIX IS ASSUMED TO BE THE IDENTITY AND
 *              NEED NOT BE DEFINED. SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
 *              IT IS ASSUMED THAT ONLY THE ELEMENTS OF RIGHT LOWER BLOCK OF
 *              DIMENSION N-M1 DIFFER FROM THAT OF THE IDENTITY MATRIX.
 *              IF (MLMAS.EQ.N-M1) THIS SUBMATRIX IS SUPPOSED TO BE FULL
 *                 AM(I,J) = M(I+M1,J+M1)     FOR I=1,N-M1 AND J=1,N-M1.
 *              ELSE, THE MASS MATRIX IS BANDED
 *                 AM(I-J+MUMAS+1,J) = M(I+M1,J+M1)
 *       - MLMAS: MLMAS=N-M1: IF THE NON-TRIVIAL PART OF M IS FULL
 *                0<=MLMAS<N-M1: LOWER BANDWIDTH OF THE MASS MATRIX
 *       - MUMAS: UPPER BANDWIDTH OF THE MASS MATRIX
 *                NEED NOT BE DEFINED IF MLMAS=N-M1
 *
 *    IWORK(9)  THE VALUE OF M1.  DEFAULT M1=0.
 *
 *    IWORK(10) THE VALUE OF M2.  DEFAULT M2=M1.
 *
 * ----------
 *
 *    WORK(1)   UROUND, THE ROUNDING UNIT, DEFAULT 1.D-16.
 *
 *    WORK(2)   THE SAFETY FACTOR IN STEP SIZE PREDICTION,
 *              DEFAULT 0.9D0.
 *
 *    WORK(3)   DECIDES WHETHER THE JACOBIAN SHOULD BE RECOMPUTED;
 *              INCREASE WORK(3), TO 0.1 SAY, WHEN JACOBIAN EVALUATIONS
 *              ARE COSTLY. FOR SMALL SYSTEMS WORK(3) SHOULD BE SMALLER
 *              (0.001D0, SAY). NEGATIV WORK(3) FORCES THE CODE TO
 *              COMPUTE THE JACOBIAN AFTER EVERY ACCEPTED STEP.
 *              DEFAULT 0.001D0.
 *
 *    WORK(4)   STOPPING CRITERION FOR NEWTON'S METHOD, USUALLY CHOSEN <1.
 *              SMALLER VALUES OF WORK(4) MAKE THE CODE SLOWER, BUT SAFER.
 *              DEFAULT MIN(0.03D0,RTOL(1)**0.5D0)
 *
 *    WORK(5) AND WORK(6) : IF WORK(5) < HNEW/HOLD < WORK(6), THEN THE
 *              STEP SIZE IS NOT CHANGED. THIS SAVES, TOGETHER WITH A
 *              LARGE WORK(3), LU-DECOMPOSITIONS AND COMPUTING TIME FOR
 *              LARGE SYSTEMS. FOR SMALL SYSTEMS ONE MAY HAVE
 *              WORK(5)=1.D0, WORK(6)=1.2D0, FOR LARGE FULL SYSTEMS
 *              WORK(5)=0.99D0, WORK(6)=2.D0 MIGHT BE GOOD.
 *              DEFAULTS WORK(5)=1.D0, WORK(6)=1.2D0 .
 *
 *    WORK(7)   MAXIMAL STEP SIZE, DEFAULT XEND-X.
 *
 *    WORK(8), WORK(9)   PARAMETERS FOR STEP SIZE SELECTION
 *              THE NEW STEP SIZE IS CHOSEN SUBJECT TO THE RESTRICTION
 *                 WORK(8) <= HNEW/HOLD <= WORK(9)
 *              DEFAULT VALUES: WORK(8)=0.2D0, WORK(9)=8.D0
 *
 *-----------------------------------------------------------------------
 *
 *     OUTPUT PARAMETERS
 *     -----------------
 *     X           X-VALUE FOR WHICH THE SOLUTION HAS BEEN COMPUTED
 *                 (AFTER SUCCESSFUL RETURN X=XEND).
 *
 *     Y(N)        NUMERICAL SOLUTION AT X
 *
 *     H           PREDICTED STEP SIZE OF THE LAST ACCEPTED STEP
 *
 *     IDID        REPORTS ON SUCCESSFULNESS UPON RETURN:
 *                   IDID= 1  COMPUTATION SUCCESSFUL,
 *                   IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)
 *                   IDID=-1  INPUT IS NOT CONSISTENT,
 *                   IDID=-2  LARGER NMAX IS NEEDED,
 *                   IDID=-3  STEP SIZE BECOMES TOO SMALL,
 *                   IDID=-4  MATRIX IS REPEATEDLY SINGULAR.
 *
 *   IWORK(14)  NFCN    NUMBER OF FUNCTION EVALUATIONS (THOSE FOR NUMERICAL
 *                      EVALUATION OF THE JACOBIAN ARE NOT COUNTED)
 *   IWORK(15)  NJAC    NUMBER OF JACOBIAN EVALUATIONS (EITHER ANALYTICALLY
 *                      OR NUMERICALLY)
 *   IWORK(16)  NSTEP   NUMBER OF COMPUTED STEPS
 *   IWORK(17)  NACCPT  NUMBER OF ACCEPTED STEPS
 *   IWORK(18)  NREJCT  NUMBER OF REJECTED STEPS (DUE TO ERROR TEST),
 *                      (STEP REJECTIONS IN THE FIRST STEP ARE NOT COUNTED)
 *   IWORK(19)  NDEC    NUMBER OF LU-DECOMPOSITIONS OF BOTH MATRICES
 *   IWORK(20)  NSOL    NUMBER OF FORWARD-BACKWARD SUBSTITUTIONS, OF BOTH
 *                      SYSTEMS; THE NSTEP FORWARD-BACKWARD SUBSTITUTIONS,
 *                      NEEDED FOR STEP SIZE SELECTION, ARE NOT COUNTED
 *-----------------------------------------------------------------------*/
void RADAU5(int* N,
            void FCN(int*, double*, double*, double*, double*, int*),
            double* X, double* Y, double* XEND, double* H,
            double* RTOL, double* ATOL, int* ITOL,
            void JAC(int*, double*, double*, double*, int*, double*, double*),
            int* IJAC, int* MLJAC, int* MUJAC,
            void MAS(int* n, double* am, int* lmas, int* rpar, int* ipar),
            int* IMAS, int* MLMAS, int* MUMAS,
            void SOLOUT(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*),
            int* IOUT,
            double* WORK, int* LWORK, int* IWORK, int* LIWORK,
            double* RPAR, int* IPAR, int* IDID);

/* C-Interface to radau5.
   Cares for memory allocation and parameter passing.
   It is possible to supply constants as parameters.
   Work and Iwork must point to arrays of at least 20 elements.
   (for Input/Output parameters only)
*/
void cradau5(int n,
             void fcn(int*, double*, double*, double*, double*, int*),
             double x, double* y, double xend, double h,
             double rtol, double atol,
             void jac(int*, double*, double*, double*, int*, double*, double*),
             int ijac, int mljac, int mujac,
             void mas(int* n, double* am, int* lmas, int* rpar, int* ipar),
             int imas, int mlmas, int mumas,
             void solout(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*),
             int iout,
             double* work, int* iwork,
             double* rpar, int* ipar, int* idid);

/* Interface to the FORTRAN function for contignuous output.(see above) */
double CONTR5(int* I, double* S, double* CONT, int* LRC);
/* C-Interface to CONTR5.
   Cares for parameter passing */
double ccontr5(int i, double s, double* cont, int* lrc);

/* FORTRAN function.
   Prints the FORTRAN interpretation of the (n,m)-matrix A */
void PRINT_MAT(int* n, int* m, double* A);

#ifdef CPP
}
#endif

#endif
