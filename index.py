from flask import Flask, render_template, request
import numpy as np
import cmath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

def mat2latex(mat):
    return r'\[\begin{bmatrix} ' + ' \\\\ '.join([' & '.join(map(str, row)) for row in mat]) + r' \end{bmatrix}\]'

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode("ascii")
    plt.close(fig)
    return img

@app.route("/", methods=["GET", "POST"])
def index():

    # ------------------ MATRICES ------------------
    A = np.array([2, 4, 6]).reshape(1,3)
    B = np.array([[1],[3]])
    C = np.array([[3, 1], [4,2]])
    D = np.array([-1,2]).reshape(1,2)
    E = np.array([[1, 0, 2], [3,1,1], [-1,2,3]])
    F = np.array([[4, 5], [1, 2]])
    G = np.array([[2], [-1], [3]])
    H = np.array([[2, 3, -1], [1,3,1], [0,2,4]])
    I = np.array([[1, -1], [2, 4]])
    K = np.array([[3, 1, 1], [2,0,1], [-1,2,4]])

    dataset = {
        'V': [F,I,C,F,I,C,F,I,F,C,F,I], 
        'X': [E, E, H, H, K, K, E, E, H, H, K, K], 
        'Y': [H, H, E,E,E,E,K,K,K,K,H,H], 
        'Z': [A,G,G,A,A,G,G,A,A,G,A,G]
    }

    MAX_LEN = len(dataset['V'])

    idx = int(request.form.get("data", 1))

    V = dataset['V'][idx-1]
    X = dataset['X'][idx-1]
    Y = dataset['Y'][idx-1]
    Z = dataset['Z'][idx-1]

    # ------------------ Q1 ------------------
    q1 = {}
    q1["a"] = X + Y
    try: q1["b"] = np.dot(X, Z)
    except: q1["b"] = np.array([["Bad","matrix"]])
    try: q1["c"] = np.dot(Z, X)
    except: q1["c"] = np.array([["Bad","matrix"]])
    q1["d"] = 3 * X
    q1["e"] = 3 * X - Y

    q1_latex = {k: mat2latex(v) for k,v in q1.items()}

    # ------------------ Q2 ------------------
    q2 = []
    for M in [V,X,Y,Z]:
        if M.shape[0] == M.shape[1]:
            q2.append((mat2latex(M), np.linalg.det(M)))

    # ------------------ Q3 ------------------
    A3 = [[2,3],[18]]
    B3 = [[1,-2],[-5]]
    C3 = [[3,1],[13]]
    D3 = [[4,-1],[8]]
    E3 = [[6,-3],[6]]
    F3 = [[5,-4],[-1]]

    eqn1 = [A3,B3,B3,B3,E3,A3,A3,B3,B3,B3,B3,C3]
    eqn2 = [F3,C3,D3,E3,F3,B3,C3,C3,D3,E3,F3,D3]

    Ax = np.array([eqn1[idx-1][0], eqn2[idx-1][0]])
    b = np.array([eqn1[idx-1][1], eqn2[idx-1][1]])
    sol = np.linalg.solve(Ax, b)
    q3_matrix = mat2latex(Ax)
    q3_rhs = mat2latex(b)
    q3_x, q3_y = sol[0][0], sol[1][0]

    # ------------------ Q4 ------------------
    A3 = [[2,-2,2],[2]]
    B3 = [[-1,-1,1],[3]]
    C3 = [[3,-1,2],[3]]
    D3 = [[-1,1,2],[11]]
    E3 = [[3,1,-2],[-9]]
    F3 = [[2,3,1],[8]]

    eq1 = [B3,B3,B3,D3,B3,D3,A3,A3,B3,B3,A3,A3]
    eq2 = [C3,C3,C3,E3,C3,E3,B3,B3,C3,C3,B3,B3]
    eq3 = [D3,E3,F3,F3,F3,F3,E3,F3,D3,E3,C3,D3]

    Ax4 = np.array([eq1[idx-1][0], eq2[idx-1][0], eq3[idx-1][0]])
    b4 = np.array([eq1[idx-1][1][0], eq2[idx-1][1][0], eq3[idx-1][1][0]])
    sol4 = np.linalg.solve(Ax4, b4)
    q4_matrix = mat2latex(Ax4)
    q4_rhs = mat2latex([[b4[0]],[b4[1]],[b4[2]]])
    a4,b4v,c4 = sol4

    # ------------------ Q5–8 ------------------
    A5, B5 = 2+3j, 3+2j
    C5, D5 = 3-2j, 1-2j
    E5, F5 = -3+2j, -2+1j

    Xc = [A5,B5,B5,B5,C5,D5,D5,D5,D5,D5,B5,B5]
    Yc = [F5,A5,C5,D5,F5,A5,B5,C5,E5,F5,E5,F5]

    Xv = Xc[idx-1]
    Yv = Yc[idx-1]

    q5a = Xv + Yv
    q5b = Yv - Xv

    Xr, Xtheta = cmath.polar(Xv)
    Yr, Ytheta = cmath.polar(Yv)
    Xr_i, Xtheta_i = cmath.polar(q5a)
    Yr_i, Ytheta_i = cmath.polar(q5b)

    fig = plt.figure()
    plt.plot(q5a.real, q5a.imag, "o")
    plt.plot(q5b.real, q5b.imag, "s")
    img_q5 = fig_to_base64(fig)

    # ------------------ Q9–11 ------------------
    A9, B9 = 3+4j, 4-5j
    C9, D9 = 5-2j, 1-3j
    E9, F9 = 2+3j, 4+1j

    Vc = [A9,B9,B9,B9,C9,D9,D9,D9,D9,D9,B9,B9]
    Wc = [F9,A9,C9,D9,F9,A9,B9,C9,E9,F9,E9,F9]

    Vv = Vc[idx-1]
    Wv = Wc[idx-1]

    q9 = Vv + Wv
    q10 = Vv - Wv

    fig2 = plt.figure()
    plt.plot(q9.real, q9.imag, "o")
    plt.plot(q10.real, q10.imag, "s")
    img_q9 = fig_to_base64(fig2)

    Z1r, Z1theta = cmath.polar(Vv)
    Z2r, Z2theta = cmath.polar(Wv)
    Zr, Ztheta = cmath.polar(q9)

    Vr, Vtheta = 12, 0
    r_vz = round(Vr / Zr,2)
    deg_vz = round(Vtheta - Ztheta*180/np.pi,2)

    return render_template(
        "index.html",
        idx=idx,
        max_len=MAX_LEN,
        mat2latex=mat2latex,
        X=X, Y=Y, Z=Z,
        q1_latex=q1_latex,
        q2=q2,
        q3_matrix=q3_matrix, q3_rhs=q3_rhs, q3_x=q3_x, q3_y=q3_y,
        q4_matrix=q4_matrix, q4_rhs=q4_rhs, a4=a4, b4=b4v, c4=c4,
        Xv=Xv, Yv=Yv, q5a=q5a, q5b=q5b,
        Xr=Xr, Xtheta=Xtheta, Yr=Yr, Ytheta=Ytheta,
        Xr_i=Xr_i, Xtheta_i=Xtheta_i, Yr_i=Yr_i, Ytheta_i=Ytheta_i,
        img_q5=img_q5,
        q9=q9, q10=q10,
        img_q9=img_q9,
        Z1r=Z1r, Z1theta=Z1theta, Z2r=Z2r, Z2theta=Z2theta,
        Zr=Zr, Ztheta=Ztheta,
        r_vz=r_vz, deg_vz=deg_vz
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
