
fn accum<const N: usize>(a: &mut [f64; N], b: &[f64; N]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a += b;
    }
}

fn softmax(a: &mut [f64]) {
    let max = a.iter().fold(f64::NAN, |a, b| a.max(*b));
    let mut sum = 0.0;
    for a in a.iter_mut() {
        *a = (*a - max).exp();
        sum += *a;
    }
    for a in a.iter_mut() {
        *a /= sum;
    }
}

fn rmsnorm<const N: usize>(o: &mut [f64; N], x: &[f64; N], w: &[f64; N]) {
    let ss = x.iter().fold(0.0, |s, n| s + n * n);
    let ms = ((ss / x.len() as f64) + 1e-5).sqrt();
    // normalize and scale
    for i in 0..o.len() {
        o[i] = x[i] * w[i] / ms;
    }
}

fn matmul<const D: usize, const N: usize>(xout: &mut [f64; D], w: &[[f64; N]; D], x: &[f64; N]) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..w.len() {
        xout[i] = 0.0;
        for j in 0..x.len() {
            xout[i] += w[i][j] * x[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accum() {
        let mut a = [1.0, 2.0];
        let b = [1.2, 3.0];
        accum(&mut a, &b);
        assert_eq!(a[0], 2.2);
        assert_eq!(a[1], 5.0);
    }

    #[test]
    fn test_matmul() {
        let w = [[1.0, 2.0, 3.0], [1.0, 5.0, 1.0]]; // (2,3)
        let x = [2.0, 4.0, 8.0]; // (3,)
        let out: &mut [f64; 2] = &mut [0.0, 0.0]; // (2, )
        matmul(out, &w, &x);
        assert_eq!(out[0], 34.0);
        assert_eq!(out[1], 30.0);
    }
}
