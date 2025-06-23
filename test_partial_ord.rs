fn test() { 
    let v1 = vec![1.0, 2.0]; 
    let v2 = vec![2.0, 1.0]; 
    let _ = v1.partial_cmp(&v2); 
    println!("Vec<f64> implements PartialOrd!");
}

fn main() { test(); }