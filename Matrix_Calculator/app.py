from flask import Flask, render_template, request


app = Flask(__name__)
def eigenvalues(matrix, iterations=1000, tolerance=1e-6):
    if len(matrix) != len(matrix[0]):
        return None  # Matrix must be square for eigenvalues
    n = len(matrix)
    v = [1.0] * n  # Initial guess for eigenvector
    lambda_prev = 0

    for _ in range(iterations):
        # Matrix-vector multiplication
        w = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
        
        # Find the largest component of w
        max_w = max(w, key=abs)
        
        # Normalize w
        v = [w[i] / max_w for i in range(n)]
        
        # Estimate eigenvalue
        lambda_est = sum(v[i] * w[i] for i in range(n))
        
        # Check for convergence
        if abs(lambda_est - lambda_prev) < tolerance:
            return lambda_est
        
        lambda_prev = lambda_est

    return None  # Eigenvalue did not converge

def eigenvectors(matrix, eigenvalue, iterations=100, tolerance=1e-6):
    if len(matrix) != len(matrix[0]):
        return ' Matrix must be square for eigenvectors'
    n = len(matrix)
    v = [1.0] * n  # Initial guess for eigenvector

    for _ in range(iterations):
        # Matrix-vector multiplication
        w = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
        
        # Find the largest component of w
        max_w = max(w, key=abs)
        
        # Normalize w
        v = [w[i] / max_w for i in range(n)]
        
        # Check for convergence
        if abs(max_w - eigenvalue) < tolerance:
            return v
        
    return v


@app.route('/')
def index():
    return render_template('matrix_calculator.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    operation = request.form['operation']
    if operation =='addition' or operation=='subtract' or operation=='multiply':


        rows1 = int(request.form['rows1'])
        cols1 = int(request.form['cols1'])
        rows2 = int(request.form['rows2'])
        cols2 = int(request.form['cols2'])

        matrix1 = []
        matrix2 = []

        for i in range(rows1):
            row1 = []
            for j in range(cols1):
                row1.append(float(request.form[f'matrix1_{i}_{j}']))
            matrix1.append(row1)
        for i in range(rows2):
            row2 = []
            for j in range(cols2):
                row2.append(float(request.form[f'matrix2_{i}_{j}']))
            matrix2.append(row2)
        if operation == 'addition':
            if rows1 != rows2 or cols1 != cols2:
                error_message = "Matrix dimensions are not compatible for addition."
                return render_template('matrix_calculator.html', error_message=error_message)
            result = [[matrix1[i][j] + matrix2[i][j] for j in range(cols1)] for i in range(rows1)]
        elif operation == 'subtract':
            if rows1 != rows2 or cols1 != cols2:
                error_message = "Matrix dimensions are not compatible for subtraction."
                return render_template('matrix_calculator.html', error_message=error_message)
            result = [[matrix1[i][j] - matrix2[i][j] for j in range(cols1)] for i in range(rows1)]
        elif operation == 'multiply':
            if cols1 != rows2:
                error_message = "Matrix dimensions are not compatible for multiplication. " \
                                "Columns of Matrix 1 must be equal to the rows of Matrix 2."
                return render_template('matrix_calculator.html', error_message=error_message)
            result = [[0] * cols2 for _ in range(rows1)]
            for i in range(rows1):
                for j in range(cols2):
                    for k in range(cols1):
                        result[i][j] += matrix1[i][k] * matrix2[k][j]
        return render_template('matrix_calculator.html', result=result, matrix1=matrix1,matrix2=matrix2, rows1=rows1, cols1=cols1, operation=operation)
    else:
        rows1 = int(request.form['rows1'])
        cols1 = int(request.form['cols1'])
        print(rows1,cols1)
        matrix1 = []
        for i in range(rows1):
            row1 = []
            for j in range(cols1):
                row1.append(float(request.form[f'matrix1_{i}_{j}']))
            matrix1.append(row1)
        if operation == 'diagonal':
            result = [[matrix1[i][j] if i == j else 0 for j in range(cols1)] for i in range(rows1)]
        elif operation == 'non_diagonal':
            result = [[matrix1[i][j] if i != j else 0 for j in range(cols1)] for i in range(rows1)]
        elif operation == 'transpose':
            result = [[matrix1[j][i] for j in range(rows1)] for i in range(cols1)]
        elif operation == 'eigenvalue':
            message = eigenvalues(matrix1)
            return render_template('matrix_calculator.html',eigenval=message)
        elif operation == 'eigenvector':
            eigenval = eigenvalues(matrix1)
            if eigenval is not None:
                eigenvector = eigenvectors(matrix1, eigenval)
                return render_template('matrix_calculator.html',eigenvector=eigenvector)
            else:
                error_message = "eigenvector cannot be calculated. " 
                return render_template('matrix_calculator.html', error_message=error_message)
        return render_template('matrix_calculator.html', result=result, matrix1=matrix1, rows1=rows1, cols1=cols1, operation=operation)

if __name__ == '__main__':
    app.run(debug=True)