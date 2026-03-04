from flask import Flask, request, render_template_string
import numpy as np
import cmath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

def mat2latex(mat):
    return r'$\begin{bmatrix} ' + ' \\\ '.join([' & '.join(map(str, row)) for row in mat]) + r' \end{bmatrix}$'

@app.route("/", methods=["GET", "POST"])
def index():
    # Define matrices
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

    # selected index
    idx = int(request.form.get("data", 1))
    V = dataset['V'][idx-1]
    X = dataset['X'][idx-1]
    Y = dataset['Y'][idx-1]
    Z = dataset['Z'][idx-1]

    # ---------- Question 1 ----------
    result = {}
    result['a) X+Y'] = X + Y
    try:
        result['b) X.Z'] = np.dot(X, Z)
    except Exception:
        result['b) X.Z'] = np.array([['Bad', 'matrix']])
    try:
        result['c) Z.X'] = np.dot(Z, X)
    except Exception:
        result['c) Z.X'] = np.array([['Bad', 'matrix']])
    result['d) 3X'] = 3 * X
    result['e) 3X-Y'] = 3 * X - Y

    q1_display = []
    for k, Ares in result.items():
        try:
            q1_display.append((k, mat2latex(Ares)))
        except Exception as e:
            q1_display.append((k, str(e)))

    # ---------- Question 2 ----------
    q2_display = []
    for item in [V,X,Y,Z]:
        if item.shape[0] == item.shape[1]:
            q2_display.append((mat2latex(item), np.linalg.det(item)))

    # ---------- Question 3 ----------
    A3 = [[2,3],[18]]
    B3 = [[1,-2],[-5]]
    C3 = [[3,1],[13]]
    D3 = [[4,-1],[8]]
    E3 = [[6,-3],[6]]
    F3 = [[5,-4],[-1]]
    eqn1 = [A3,B3,B3,B3,E3,A3,A3,B3,B3,B3,B3,C3]
    eqn2 = [F3,C3,D3,E3,F3,B3,C3,C3,D3,E3,F3,D3]

    def solve_eqn2(e1,e2):
        Ax = np.array([e1[0],e2[0]])
        b = np.array([e1[1],e2[1]])
        soln = np.linalg.solve(Ax, b)
        return Ax, b, soln

    Ax3, b3, sol3 = solve_eqn2(eqn1[idx-1], eqn2[idx-1])
    q3_matrix = mat2latex(Ax3)
    q3_rhs = mat2latex(b3)
    q3_x, q3_y = sol3[0][0], sol3[1][0]

    # ---------- Question 4 ----------
    A3 = [[2,-2,2],[2]]
    B3 = [[-1,-1,1],[3]]
    C3 = [[3,-1,2],[3]]
    D3 = [[-1,1,2],[11]]
    E3 = [[3,1,-2],[-9]]
    F3 = [[2,3,1],[8]]

    eqn1_3 = [B3,B3,B3,D3,B3,D3,A3,A3,B3,B3,A3,A3]
    eqn2_3 = [C3,C3,C3,E3,C3,E3,B3,B3,C3,C3,B3,B3]
    eqn3_3 = [D3,E3,F3,F3,F3,F3,E3,F3,D3,E3,C3,D3]

    def solve_eqn3(e1,e2,e3):
        Ax = np.array([e1[0],e2[0],e3[0]])
        constants = [e1[1][0], e2[1][0], e3[1][0]]
        b = np.array(constants)
        soln = np.linalg.solve(Ax, b)
        return Ax, constants, soln

    Ax4, const4, sol4 = solve_eqn3(eqn1_3[idx-1], eqn2_3[idx-1], eqn3_3[idx-1])
    q4_matrix = mat2latex(Ax4)
    q4_rhs = mat2latex([[const4[0]],[const4[1]],[const4[2]]])
    a4, b4, c4 = round(sol4[0],1), round(sol4[1],1), round(sol4[2],1)

    # ---------- Question 5–8 ----------
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

    # plot for Q5
    fig1, ax1 = plt.subplots()
    ax1.plot(q5a.real, q5a.imag, 'o')
    ax1.plot(q5b.real, q5b.imag, 's')
    ax1.text(q5a.real+0.01,q5a.imag-0.1,f"Q5a {q5a.real,q5a.imag}", color="g")
    ax1.text(q5b.real-0.6,q5b.imag+0.1,f"Q5b {q5b}", color='blue')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", bbox_inches="tight")
    buf1.seek(0)
    img1 = base64.b64encode(buf1.getvalue()).decode("ascii")
    plt.close(fig1)

    r_mul = Xr*Yr
    deg_mul = Xtheta*180/np.pi + Ytheta*180/np.pi
    r_div = Xr/Yr
    deg_div = Xtheta*180/np.pi - Ytheta*180/np.pi

    # ---------- Question 9–11 ----------
    A9, B9 = 3+4j, 4-5j
    C9, D9 = 5-2j, 1-3j
    E9, F9 = 2+3j, 4+1j
    Vc = [A9,B9,B9,B9,C9,D9,D9,D9,D9,D9,B9,B9]
    Wc = [F9,A9,C9,D9,F9,A9,B9,C9,E9,F9,E9,F9]
    Vv = Vc[idx-1]
    Wv = Wc[idx-1]
    q9 = Vv + Wv
    q10 = Vv - Wv

    fig2, ax2 = plt.subplots()
    ax2.plot(q9.real, q9.imag, 'o')
    ax2.text(q9.real-1.3,q9.imag-0.1,f"Q9 {q9}", color="g")
    ax2.plot(q10.real, q10.imag, 's')
    ax2.text(q10.real+0.01,q10.imag+0.1,f"Q10 {q10}", color='blue')
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", bbox_inches="tight")
    buf2.seek(0)
    img2 = base64.b64encode(buf2.getvalue()).decode("ascii")
    plt.close(fig2)

    Z1r, Z1theta = cmath.polar(Vv)
    Z2r, Z2theta = cmath.polar(Wv)
    Zr, Ztheta = cmath.polar(q9)
    Vr, Vtheta = 12, 0
    r_vz = Vr/Zr
    deg_vz = Vtheta - Ztheta*180/np.pi

    html = """
    <!doctype html>
    <html>
    <head>
        <title>FMath A2 - Solution (Flask)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { border: 1px solid #ccc; padding: 10px; margin-bottom: 15px; }
            h2 { margin-top: 0; }
        </style>
    </head>
    <body>
        <h1>FMath A2 - Solution (Flask)</h1>
        <form method="post">
            <label>Select Dataset:</label>
            <select name="data" onchange="this.form.submit()">
                {% for i in range(1, max_len+1) %}
                    <option value="{{i}}" {% if i == idx %}selected{% endif %}>{{i}}</option>
                {% endfor %}
            </select>
            <noscript><button type="submit">Go</button></noscript>
        </form>

        <div class="section">
            <h2>Question 1</h2>
            <p>X = {{ mat2latex(X) }} &nbsp;&nbsp; Y = {{ mat2latex(Y) }} &nbsp;&nbsp; Z = {{ mat2latex(Z) }}</p>
            <hr>
            {% for label, mat in q1_display %}
                <p>{{label}} = {{mat}}</p>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Question 2</h2>
            {% for m, det in q2_display %}
                <p>{{m}} determinant = {{det}}</p>
            {% endfor %}
        </div>

        <div class="section">
            <h2>Question 3</h2>
            <p>{{q3_matrix}} [x y]^T = {{q3_rhs}}</p>
            <p>x = {{q3_x}}, y = {{q3_y}}</p>
        </div>

        <div class="section">
            <h2>Question 4</h2>
            <p>{{q4_matrix}} [a b c]^T = {{q4_rhs}}</p>
            <p>a = {{a4}}, b = {{b4}}, c = {{c4}}</p>
        </div>

        <div class="section">
            <h2>Questions 5,6,7,8 (Complex)</h2>
            <p>X = {{Xv}}, Y = {{Yv}}</p>
            <p>5a) X + Y = {{q5a}}</p>
            <p>5b) Y - X = {{q5b}}</p>
            <p>6a) X_polar = {{ "%.2f" % Xr }}&lt;{{ "%.2f" % (Xtheta*180/3.14159) }}</p>
            <p>6b) Y_polar = {{ "%.2f" % Yr }}&lt;{{ "%.2f" % (Ytheta*180/3.14159) }}</p>
            <p>6ai) X+Y_polar = {{ "%.2f" % Xr_i }}&lt;{{ "%.2f" % (Xtheta_i*180/3.14159) }}</p>
            <p>6bi) Y-X_polar = {{ "%.2f" % Yr_i }}&lt;{{ "%.2f" % (Ytheta_i*180/3.14159) }}</p>
            <img src="data:image/png;base64,{{img1}}" alt="Q5 plot">
            <p>7a) X·Y = {{ "%.2f" % r_mul }}&lt;{{ "%.2f" % deg_mul }}</p>
            <p>7b) X/Y = {{ "%.2f" % r_div }}&lt;{{ "%.2f" % deg_div }}</p>
            <p>8a) cartesian: X·Y = {{ "%.2f" % (Xv*Yv.real + 0j + Xv*Yv.imag*0j).real if False else "%.2f" % (Xv*Yv).real }}</p>
            <p>8b) cartesian: X/Y = {{ "%.2f" % (Xv/Yv).real }}</p>
        </div>

        <div class="section">
            <h2>Questions 9,10,11 (Complex)</h2>
            <p>V = {{Vv}}, W = {{Wv}}</p>
            <p>9) V + W = {{q9}}</p>
            <p>10) V - W = {{q10}}</p>
            <img src="data:image/png;base64,{{img2}}" alt="Q9-10 plot">
            <p>11) V_rect = {{ "%.2f" % Z1r }}&lt;{{ "%.2f" % (Z1theta*180/3.14159) }}</p>
            <p>11) W_rect = {{ "%.2f" % Z2r }}&lt;{{ "%.2f" % (Z2theta*180/3.14159) }}</p>
            <p>11) Z_Trect = {{q9}}</p>
            <p>11) Z_Tpolar = {{ "%.2f" % Zr }}&lt;{{ "%.2f" % (Ztheta*180/3.14159) }}</p>
            <p>11) V_polar = 12&lt;0</p>
            <p>11) V/Z = {{ "%.2f" % r_vz }}&lt;{{ "%.2f" % deg_vz }}</p>
        </div>
    </body>
    </html>
    """

    return render_template_string(
        html,
        max_len=MAX_LEN,
        idx=idx,
        mat2latex=mat2latex,
        X=X, Y=Y, Z=Z,
        q1_display=q1_display,
        q2_display=q2_display,
        q3_matrix=q3_matrix, q3_rhs=q3_rhs, q3_x=q3_x, q3_y=q3_y,
        q4_matrix=q4_matrix, q4_rhs=q4_rhs, a4=a4, b4=b4, c4=c4,
        Xv=Xv, Yv=Yv, q5a=q5a, q5b=q5b,
        Xr=Xr, Xtheta=Xtheta, Yr=Yr, Ytheta=Ytheta,
        Xr_i=Xr_i, Xtheta_i=Xtheta_i, Yr_i=Yr_i, Ytheta_i=Ytheta_i,
        img1=img1,
        r_mul=r_mul, deg_mul=deg_mul, r_div=r_div, deg_div=deg_div,
        Vv=Vv, Wv=Wv, q9=q9, q10=q10,
        img2=img2,
        Z1r=Z1r, Z1theta=Z1theta, Z2r=Z2r, Z2theta=Z2theta,
        Zr=Zr, Ztheta=Ztheta,
        r_vz=r_vz, deg_vz=deg_vz
    )

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
