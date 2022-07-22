#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "roccor/RoccoR.h"

/*
 *
 */

RoccoR rc("roccor/RoccoR2017UL.txt");

static PyMethodDef Methods[] = {
        {NULL, NULL, 0, NULL}
};

static void kScaleDT(char **args, npy_intp const* dimensions,
                            npy_intp const* steps, void* data){
    npy_intp i;
    npy_intp n = dimensions[0];
    char *QIN = args[0], *ptIN = args[1], *etaIN = args[2],
         *phiIN = args[3], *sIN = args[4], *mIN = args[5], 
         *out = args[6];

    npy_intp Q_step = steps[0], pt_step = steps[1],
             eta_step = steps[2], phi_step = steps[3],
             s_step = steps[4], m_step = steps[5],
             out_step = steps[6];

    int Q, s, m;
    double pt, eta, phi;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        Q = *(int *)QIN;
        pt = *(double *)ptIN;
        eta = *(double *)etaIN;
        phi = *(double *)phiIN;
        s = *(int *)sIN;
        m = *(int *)mIN;

        *((double *)out) = rc.kScaleDT(Q, pt, eta, phi, s, m);

        QIN += Q_step;
        ptIN += pt_step;
        etaIN += eta_step;
        phiIN += phi_step;
        sIN += s_step;
        mIN += m_step;
        out += out_step;
    }
}

static void kSpreadMC(char **args, npy_intp const* dimensions,
                            npy_intp const* steps, void* data){
    npy_intp i;
    npy_intp n = dimensions[0];
    char *QIN = args[0], *ptIN = args[1], *etaIN = args[2],
         *phiIN = args[3], *genPtIN = args[4], *sIN = args[5], 
         *mIN = args[6], *out = args[7];

    npy_intp Q_step = steps[0], pt_step = steps[1],
             eta_step = steps[2], phi_step = steps[3],
             genPt_step = steps[4], s_step = steps[5], 
             m_step = steps[6], out_step = steps[7];

    int Q, s, m;
    double pt, eta, phi, genPt;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        Q = *(int *)QIN;
        pt = *(double *)ptIN;
        eta = *(double *)etaIN;
        phi = *(double *)phiIN;
        genPt = *(double *)genPtIN;
        s = *(int *)sIN;
        m = *(int *)mIN;

        *((double *)out) = rc.kSpreadMC(Q, pt, eta, phi, genPt, s, m);

        QIN += Q_step;
        ptIN += pt_step;
        etaIN += eta_step;
        phiIN += phi_step;
        genPtIN += genPt_step;
        sIN += s_step;
        mIN += m_step;
        out += out_step;
    }
}

static void kSmearMC(char **args, npy_intp const* dimensions,
                            npy_intp const* steps, void* data){
    npy_intp i;
    npy_intp n = dimensions[0];
    char *QIN = args[0], *ptIN = args[1], *etaIN = args[2],
         *phiIN = args[3], *nlIN = args[4], *uIN = args[5],
         *sIN = args[6], *mIN = args[7], *out = args[8];

    npy_intp Q_step = steps[0], pt_step = steps[1],
             eta_step = steps[2], phi_step = steps[3],
             nl_step = steps[4], u_step = steps[5],
             s_step = steps[6], m_step = steps[7], 
             out_step = steps[8];

    int Q, s, m, nl;
    double pt, eta, phi, u;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        Q = *(int *)QIN;
        pt = *(double *)ptIN;
        eta = *(double *)etaIN;
        phi = *(double *)phiIN;
        nl = *(int *)nlIN;
        u = *(double *)uIN;
        s = *(int *)sIN;
        m = *(int *)mIN;

        *((double *)out) = rc.kSmearMC(Q, pt, eta, phi, nl, u, s, m);

        QIN += Q_step;
        ptIN += pt_step;
        etaIN += eta_step;
        phiIN += phi_step;
        nlIN += nl_step;
        uIN += u_step;
        sIN += s_step;
        mIN += m_step;
        out += out_step;
    }
}

/* function pointers */
PyUFuncGenericFunction funcs_kScaleDT[1] = {&kScaleDT};
PyUFuncGenericFunction funcs_kSpreadMC[1] = {&kSpreadMC};
PyUFuncGenericFunction funcs_kSmearMC[1] = {&kSmearMC};

/* data types */
static char types_kScaleDT[7] = {NPY_INT64, NPY_DOUBLE,
                                NPY_DOUBLE, NPY_DOUBLE,
                                NPY_INT64, NPY_INT64,
                                NPY_DOUBLE};
static char types_kSpreadMC[8]= {NPY_INT64, NPY_DOUBLE,
                                NPY_DOUBLE, NPY_DOUBLE,
                                NPY_DOUBLE, NPY_INT64,
                                NPY_INT64, NPY_DOUBLE};
static char types_kSmearMC[9] = {NPY_INT64, NPY_DOUBLE,
                                NPY_DOUBLE, NPY_DOUBLE,
                                NPY_INT64, NPY_DOUBLE,
                                NPY_INT64, NPY_INT64,
                                NPY_DOUBLE};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "roccor",
    NULL,
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_roccor(void)
{
    PyObject *m, *kScaleDT, *kSpreadMC, *kSmearMC, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    kScaleDT = PyUFunc_FromFuncAndData(funcs_kScaleDT, data, types_kScaleDT, 1, 6, 1,
                                    PyUFunc_None, "kScaleDT",
                                    "kScaleDT_docstring", 0);
    kSpreadMC = PyUFunc_FromFuncAndData(funcs_kSpreadMC, data, types_kSpreadMC, 1, 7, 1,
                                    PyUFunc_None, "kSpreadMC",
                                    "kSpreadMC_docstring", 0);
    kSmearMC = PyUFunc_FromFuncAndData(funcs_kSmearMC, data, types_kSmearMC, 1, 8, 1,
                                    PyUFunc_None, "kSmearMC",
                                    "kSmearMC_docstring", 0);


    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "kScaleDT", kScaleDT);
    PyDict_SetItemString(d, "kSpreadMC", kSpreadMC);
    PyDict_SetItemString(d, "kSmearMC", kSmearMC);

    Py_DECREF(kScaleDT);
    Py_DECREF(kSpreadMC);
    Py_DECREF(kSmearMC);

    return m;
}
