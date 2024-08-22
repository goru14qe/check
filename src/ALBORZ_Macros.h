#ifndef ALBORZ_MACROS_H_
#define ALBORZ_MACROS_H_

// Logical constants
#define TRUE 1
#define FALSE -1

template <typename T>
T sqr(T x) {
	return x * x;
}

template <typename T1, typename T2>
auto dot(const T1& a, const T2& b) -> decltype(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// Macro functions
#define SQ(x) ((x) * (x))  // square function; replaces SQ(x) by ((x) * (x)) in the code
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MULTIPLY(a, b) ((a) * (b))
#define DOT(a, b) (MULTIPLY((a)[0], (b)[0]) + MULTIPLY((a)[1], (b)[1]) + MULTIPLY((a)[2], (b)[2]))
#define MAG(a) (SQ((a)[0]) + SQ((a)[1]) + SQ((a)[2]))
#define SGN(a) ((a > 0) ? 1 : ((a < 0) ? -1 : 0))

#define POSITIVE(a) ((a) < 0 ? 0 : (a))
#define NEGATIVE(a) ((a) > 0 ? 0 : (a))

#define M_PI 3.14159265358979323846 /* pi */
#define R_GAS 8.3144598             /* units: kg.m2/s2.K.mol, or J/K.mol */
//       MPI
#define MASTER 0       // processor_id of first task
#define FROM_MASTER 1  // setting a message type
#define FROM_WORKER 2  // setting a message type
#define LTAG 3         /* message tag */
#define RTAG 4         /* message tag */
#define MTAG 5         /* message tag */
#define PTAG 6         /* message tag */
//#define NONE -1        /* indicates no neighbor */
#endif
