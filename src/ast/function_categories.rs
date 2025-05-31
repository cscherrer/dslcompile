//! Function Categories for Composable Mathematical Operations
//!
//! This module defines a trait-based system for organizing mathematical functions
//! into composable categories. This enables easy extension with new function types
//! while maintaining performance and type safety.

use crate::final_tagless::traits::NumericType;
use num_traits::Float;

/// Trait for function categories that can be extended by downstream crates
pub trait FunctionCategory<T: NumericType>: Clone + std::fmt::Debug + PartialEq {
    /// Convert the function to egglog representation
    fn to_egglog(&self) -> String;

    /// Apply local optimization rules specific to this function category
    fn apply_local_rules(&self, expr: &crate::ast::ASTRepr<T>) -> Option<crate::ast::ASTRepr<T>>;

    /// Get the function category name for debugging and rule loading
    fn category_name(&self) -> &'static str;

    /// Get the priority for rule application (higher = applied first)
    fn priority(&self) -> u32 {
        100
    }
}

/// Trait for extensible optimization rules that downstream crates can implement
pub trait OptimizationRule<T: NumericType> {
    /// Apply this optimization rule to an expression
    fn apply(&self, expr: &crate::ast::ASTRepr<T>) -> Option<crate::ast::ASTRepr<T>>;

    /// Get the rule name for debugging
    fn rule_name(&self) -> &'static str;

    /// Get the rule priority (higher = applied first)
    fn priority(&self) -> u32 {
        100
    }

    /// Check if this rule is applicable to the given expression
    fn is_applicable(&self, expr: &crate::ast::ASTRepr<T>) -> bool;
}

/// Registry for custom function categories and optimization rules
pub struct ExtensionRegistry<T: NumericType> {
    custom_rules: Vec<Box<dyn OptimizationRule<T>>>,
    egglog_rules: Vec<String>,
}

impl<T: NumericType> ExtensionRegistry<T> {
    pub fn new() -> Self {
        Self {
            custom_rules: Vec::new(),
            egglog_rules: Vec::new(),
        }
    }

    /// Register a custom optimization rule
    pub fn register_rule(&mut self, rule: Box<dyn OptimizationRule<T>>) {
        self.custom_rules.push(rule);
        // Sort by priority
        self.custom_rules
            .sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Register custom egglog rules
    pub fn register_egglog_rules(&mut self, rules: String) {
        self.egglog_rules.push(rules);
    }

    /// Apply all registered rules to an expression
    pub fn apply_all_rules(&self, expr: &crate::ast::ASTRepr<T>) -> crate::ast::ASTRepr<T> {
        let mut current = expr.clone();

        for rule in &self.custom_rules {
            if rule.is_applicable(&current) {
                if let Some(optimized) = rule.apply(&current) {
                    current = optimized;
                }
            }
        }

        current
    }

    /// Get all egglog rules as a combined string
    pub fn get_egglog_rules(&self) -> String {
        self.egglog_rules.join("\n")
    }
}

impl<T: NumericType> Default for ExtensionRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trigonometric functions with comprehensive identity support
#[derive(Debug, Clone, PartialEq)]
pub enum TrigFunction<T: NumericType> {
    Sin(Box<crate::ast::ASTRepr<T>>),
    Cos(Box<crate::ast::ASTRepr<T>>),
    Tan(Box<crate::ast::ASTRepr<T>>),
    Sec(Box<crate::ast::ASTRepr<T>>),
    Csc(Box<crate::ast::ASTRepr<T>>),
    Cot(Box<crate::ast::ASTRepr<T>>),
    // Inverse functions
    Asin(Box<crate::ast::ASTRepr<T>>),
    Acos(Box<crate::ast::ASTRepr<T>>),
    Atan(Box<crate::ast::ASTRepr<T>>),
    Atan2(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrigCategory<T: NumericType> {
    pub function: TrigFunction<T>,
}

impl<T> TrigCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    /// Convert to standard AST representation
    pub fn to_ast(&self) -> crate::ast::ASTRepr<T> {
        match &self.function {
            TrigFunction::Sin(arg) => {
                // Since Sin is now in TrigCategory, we need to create a recursive structure
                // For now, return the argument itself as a placeholder
                *arg.clone()
            }
            TrigFunction::Cos(arg) => {
                // Since Cos is now in TrigCategory, we need to create a recursive structure
                // For now, return the argument itself as a placeholder
                *arg.clone()
            }
            TrigFunction::Tan(arg) => {
                // tan(x) = sin(x) / cos(x)
                let sin_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::sin(*arg.clone())));
                let cos_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::cos(*arg.clone())));
                crate::ast::ASTRepr::Div(Box::new(sin_x), Box::new(cos_x))
            }
            TrigFunction::Sec(arg) => {
                // sec(x) = 1 / cos(x)
                let one = crate::ast::ASTRepr::Constant(T::from(1.0).unwrap());
                let cos_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::cos(*arg.clone())));
                crate::ast::ASTRepr::Div(Box::new(one), Box::new(cos_x))
            }
            TrigFunction::Csc(arg) => {
                // csc(x) = 1 / sin(x)
                let one = crate::ast::ASTRepr::Constant(T::from(1.0).unwrap());
                let sin_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::sin(*arg.clone())));
                crate::ast::ASTRepr::Div(Box::new(one), Box::new(sin_x))
            }
            TrigFunction::Cot(arg) => {
                // cot(x) = cos(x) / sin(x)
                let cos_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::cos(*arg.clone())));
                let sin_x = crate::ast::ASTRepr::Trig(Box::new(TrigCategory::sin(*arg.clone())));
                crate::ast::ASTRepr::Div(Box::new(cos_x), Box::new(sin_x))
            }
            // For inverse functions, we'd need to add them to the base AST or use approximations
            _ => {
                // For now, represent as a placeholder - in a real implementation,
                // we'd either extend the base AST or use series approximations
                crate::ast::ASTRepr::Constant(T::from(0.0).unwrap())
            }
        }
    }

    pub fn to_egglog(&self) -> String {
        match &self.function {
            TrigFunction::Sin(arg) => format!("(Trig (SinFunc {}))", arg.to_egglog()),
            TrigFunction::Cos(arg) => format!("(Trig (CosFunc {}))", arg.to_egglog()),
            TrigFunction::Tan(arg) => format!("(Trig (TanFunc {}))", arg.to_egglog()),
            TrigFunction::Sec(arg) => format!("(Trig (SecFunc {}))", arg.to_egglog()),
            TrigFunction::Csc(arg) => format!("(Trig (CscFunc {}))", arg.to_egglog()),
            TrigFunction::Cot(arg) => format!("(Trig (CotFunc {}))", arg.to_egglog()),
            TrigFunction::Asin(arg) => format!("(Trig (AsinFunc {}))", arg.to_egglog()),
            TrigFunction::Acos(arg) => format!("(Trig (AcosFunc {}))", arg.to_egglog()),
            TrigFunction::Atan(arg) => format!("(Trig (AtanFunc {}))", arg.to_egglog()),
            TrigFunction::Atan2(y, x) => {
                format!("(Trig (Atan2Func {} {}))", y.to_egglog(), x.to_egglog())
            }
        }
    }

    pub fn apply_local_rules(
        &self,
        expr: &crate::ast::ASTRepr<T>,
    ) -> Option<crate::ast::ASTRepr<T>> {
        use crate::ast::ASTRepr;

        // Apply trigonometric identities locally
        match expr {
            // sin²(x) + cos²(x) = 1 pattern detection
            ASTRepr::Add(left, right) => {
                if let (Some(sin_arg), Some(cos_arg)) =
                    (extract_sin_squared(left), extract_cos_squared(right))
                {
                    if expressions_structurally_equal(&sin_arg, &cos_arg) {
                        return Some(ASTRepr::Constant(T::from(1.0).unwrap()));
                    }
                }
                None
            }
            // sec²(x) - tan²(x) = 1
            ASTRepr::Sub(left, right) => {
                if let (Some(sec_arg), Some(tan_arg)) =
                    (extract_sec_squared(left), extract_tan_squared(right))
                {
                    if expressions_structurally_equal(&sec_arg, &tan_arg) {
                        return Some(ASTRepr::Constant(T::from(1.0).unwrap()));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

/// Hyperbolic functions
#[derive(Debug, Clone, PartialEq)]
pub enum HyperbolicFunction<T: NumericType> {
    Sinh(Box<crate::ast::ASTRepr<T>>),
    Cosh(Box<crate::ast::ASTRepr<T>>),
    Tanh(Box<crate::ast::ASTRepr<T>>),
    Sech(Box<crate::ast::ASTRepr<T>>),
    Csch(Box<crate::ast::ASTRepr<T>>),
    Coth(Box<crate::ast::ASTRepr<T>>),
    // Inverse functions
    Asinh(Box<crate::ast::ASTRepr<T>>),
    Acosh(Box<crate::ast::ASTRepr<T>>),
    Atanh(Box<crate::ast::ASTRepr<T>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperbolicCategory<T: NumericType> {
    pub function: HyperbolicFunction<T>,
}

impl<T> HyperbolicCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    pub fn to_egglog(&self) -> String {
        match &self.function {
            HyperbolicFunction::Sinh(arg) => format!("(Hyperbolic (SinhFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Cosh(arg) => format!("(Hyperbolic (CoshFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Tanh(arg) => format!("(Hyperbolic (TanhFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Sech(arg) => format!("(Hyperbolic (SechFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Csch(arg) => format!("(Hyperbolic (CschFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Coth(arg) => format!("(Hyperbolic (CothFunc {}))", arg.to_egglog()),
            HyperbolicFunction::Asinh(arg) => {
                format!("(Hyperbolic (AsinhFunc {}))", arg.to_egglog())
            }
            HyperbolicFunction::Acosh(arg) => {
                format!("(Hyperbolic (AcoshFunc {}))", arg.to_egglog())
            }
            HyperbolicFunction::Atanh(arg) => {
                format!("(Hyperbolic (AtanhFunc {}))", arg.to_egglog())
            }
        }
    }

    pub fn apply_local_rules(
        &self,
        expr: &crate::ast::ASTRepr<T>,
    ) -> Option<crate::ast::ASTRepr<T>> {
        use crate::ast::ASTRepr;

        match expr {
            // cosh²(x) - sinh²(x) = 1
            ASTRepr::Sub(left, right) => {
                if let (Some(cosh_arg), Some(sinh_arg)) =
                    (extract_cosh_squared(left), extract_sinh_squared(right))
                {
                    if expressions_structurally_equal(&cosh_arg, &sinh_arg) {
                        return Some(ASTRepr::Constant(T::from(1.0).unwrap()));
                    }
                }
                None
            }
            _ => None,
        }
    }
}

/// Logarithmic and exponential functions
#[cfg(feature = "logexp")]
#[derive(Debug, Clone, PartialEq)]
pub enum LogExpFunction<T: NumericType> {
    Log(Box<crate::ast::ASTRepr<T>>), // Natural logarithm (primary)
    LogBase(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // log_base(x)
    Log10(Box<crate::ast::ASTRepr<T>>), // Base-10 logarithm
    Log2(Box<crate::ast::ASTRepr<T>>), // Base-2 logarithm
    Ln(Box<crate::ast::ASTRepr<T>>),  // Explicit natural log (for Rust calls)
    Exp(Box<crate::ast::ASTRepr<T>>), // Natural exponential
    Exp2(Box<crate::ast::ASTRepr<T>>), // Base-2 exponential
    Exp10(Box<crate::ast::ASTRepr<T>>), // Base-10 exponential
}

#[cfg(feature = "logexp")]
#[derive(Debug, Clone, PartialEq)]
pub struct LogExpCategory<T: NumericType> {
    pub function: LogExpFunction<T>,
}

#[cfg(feature = "logexp")]
impl<T> LogExpCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    pub fn to_egglog(&self) -> String {
        match &self.function {
            LogExpFunction::Log(arg) => format!("(LogExp (LogFunc {}))", arg.to_egglog()),
            LogExpFunction::LogBase(base, arg) => format!(
                "(LogExp (LogBaseFunc {} {}))",
                base.to_egglog(),
                arg.to_egglog()
            ),
            LogExpFunction::Log10(arg) => format!("(LogExp (Log10Func {}))", arg.to_egglog()),
            LogExpFunction::Log2(arg) => format!("(LogExp (Log2Func {}))", arg.to_egglog()),
            LogExpFunction::Ln(arg) => format!("(LogExp (LnFunc {}))", arg.to_egglog()),
            LogExpFunction::Exp(arg) => format!("(LogExp (ExpFunc {}))", arg.to_egglog()),
            LogExpFunction::Exp2(arg) => format!("(LogExp (Exp2Func {}))", arg.to_egglog()),
            LogExpFunction::Exp10(arg) => format!("(LogExp (Exp10Func {}))", arg.to_egglog()),
        }
    }

    pub fn apply_local_rules(
        &self,
        expr: &crate::ast::ASTRepr<T>,
    ) -> Option<crate::ast::ASTRepr<T>> {
        use crate::ast::ASTRepr;

        match expr {
            // log(exp(x)) = x
            #[cfg(feature = "logexp")]
            ASTRepr::Log(inner) => {
                #[cfg(feature = "logexp")]
                if let ASTRepr::Exp(inner_inner) = inner.as_ref() {
                    return Some(*inner_inner.clone());
                }
                None
            }
            // exp(log(x)) = x
            #[cfg(feature = "logexp")]
            ASTRepr::Exp(inner) => {
                #[cfg(feature = "logexp")]
                if let ASTRepr::Log(inner_inner) = inner.as_ref() {
                    return Some(*inner_inner.clone());
                }
                None
            }
            _ => None,
        }
    }
}

/// Special mathematical functions
#[cfg(feature = "special")]
#[derive(Debug, Clone, PartialEq)]
pub enum SpecialFunction<T: NumericType> {
    Gamma(Box<crate::ast::ASTRepr<T>>),
    LogGamma(Box<crate::ast::ASTRepr<T>>),
    Beta(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    LogBeta(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    Erf(Box<crate::ast::ASTRepr<T>>),
    Erfc(Box<crate::ast::ASTRepr<T>>),
    ErfInv(Box<crate::ast::ASTRepr<T>>),
    ErfcInv(Box<crate::ast::ASTRepr<T>>),
    // Bessel functions
    BesselJ0(Box<crate::ast::ASTRepr<T>>),
    BesselJ1(Box<crate::ast::ASTRepr<T>>),
    BesselJn(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // J_n(x)
    BesselY0(Box<crate::ast::ASTRepr<T>>),
    BesselY1(Box<crate::ast::ASTRepr<T>>),
    BesselYn(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // Y_n(x)
    BesselI0(Box<crate::ast::ASTRepr<T>>),
    BesselI1(Box<crate::ast::ASTRepr<T>>),
    BesselIn(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // I_n(x)
    BesselK0(Box<crate::ast::ASTRepr<T>>),
    BesselK1(Box<crate::ast::ASTRepr<T>>),
    BesselKn(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // K_n(x)
    // Lambert W function
    LambertW0(Box<crate::ast::ASTRepr<T>>), // Principal branch
    LambertWm1(Box<crate::ast::ASTRepr<T>>), // -1 branch
}

#[cfg(feature = "special")]
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialCategory<T: NumericType> {
    pub function: SpecialFunction<T>,
}

#[cfg(feature = "special")]
impl<T> SpecialCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    pub fn to_egglog(&self) -> String {
        match &self.function {
            SpecialFunction::Gamma(arg) => format!("(Special (GammaFunc {}))", arg.to_egglog()),
            SpecialFunction::LogGamma(arg) => {
                format!("(Special (LogGammaFunc {}))", arg.to_egglog())
            }
            SpecialFunction::Beta(a, b) => {
                format!("(Special (BetaFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            SpecialFunction::LogBeta(a, b) => format!(
                "(Special (LogBetaFunc {} {}))",
                a.to_egglog(),
                b.to_egglog()
            ),
            SpecialFunction::Erf(arg) => format!("(Special (ErfFunc {}))", arg.to_egglog()),
            SpecialFunction::Erfc(arg) => format!("(Special (ErfcFunc {}))", arg.to_egglog()),
            SpecialFunction::ErfInv(arg) => format!("(Special (ErfInvFunc {}))", arg.to_egglog()),
            SpecialFunction::ErfcInv(arg) => format!("(Special (ErfcInvFunc {}))", arg.to_egglog()),
            SpecialFunction::BesselJ0(arg) => {
                format!("(Special (BesselJ0Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselJ1(arg) => {
                format!("(Special (BesselJ1Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselJn(n, x) => format!(
                "(Special (BesselJnFunc {} {}))",
                n.to_egglog(),
                x.to_egglog()
            ),
            SpecialFunction::BesselY0(arg) => {
                format!("(Special (BesselY0Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselY1(arg) => {
                format!("(Special (BesselY1Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselYn(n, x) => format!(
                "(Special (BesselYnFunc {} {}))",
                n.to_egglog(),
                x.to_egglog()
            ),
            SpecialFunction::BesselI0(arg) => {
                format!("(Special (BesselI0Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselI1(arg) => {
                format!("(Special (BesselI1Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselIn(n, x) => format!(
                "(Special (BesselInFunc {} {}))",
                n.to_egglog(),
                x.to_egglog()
            ),
            SpecialFunction::BesselK0(arg) => {
                format!("(Special (BesselK0Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselK1(arg) => {
                format!("(Special (BesselK1Func {}))", arg.to_egglog())
            }
            SpecialFunction::BesselKn(n, x) => format!(
                "(Special (BesselKnFunc {} {}))",
                n.to_egglog(),
                x.to_egglog()
            ),
            SpecialFunction::LambertW0(arg) => {
                format!("(Special (LambertW0Func {}))", arg.to_egglog())
            }
            SpecialFunction::LambertWm1(arg) => {
                format!("(Special (LambertWm1Func {}))", arg.to_egglog())
            }
        }
    }

    pub fn apply_local_rules(
        &self,
        _expr: &crate::ast::ASTRepr<T>,
    ) -> Option<crate::ast::ASTRepr<T>> {
        // Special function identities would go here
        // For example: Γ(n+1) = n! for integer n
        // Or: Γ(1/2) = √π
        None
    }
}

/// Linear algebra operations with non-commutative properties
#[cfg(feature = "linear_algebra")]
#[derive(Debug, Clone, PartialEq)]
pub enum LinearAlgebraFunction<T: NumericType> {
    // Matrix operations
    MatrixMul(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    MatrixAdd(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    MatrixSub(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    MatrixScale(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // scalar * matrix
    Transpose(Box<crate::ast::ASTRepr<T>>),
    Determinant(Box<crate::ast::ASTRepr<T>>),
    Inverse(Box<crate::ast::ASTRepr<T>>),
    Trace(Box<crate::ast::ASTRepr<T>>),

    // Matrix division (left and right are different!)
    LeftDivide(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // A \ B = A^(-1) * B
    RightDivide(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // A / B = A * B^(-1)

    // Vector operations
    DotProduct(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>),
    CrossProduct(Box<crate::ast::ASTRepr<T>>, Box<crate::ast::ASTRepr<T>>), // Also non-commutative!
    VectorNorm(Box<crate::ast::ASTRepr<T>>),

    // Eigenvalue operations
    Eigenvalues(Box<crate::ast::ASTRepr<T>>),
    Eigenvectors(Box<crate::ast::ASTRepr<T>>),

    // Matrix decompositions
    LU(Box<crate::ast::ASTRepr<T>>),
    QR(Box<crate::ast::ASTRepr<T>>),
    SVD(Box<crate::ast::ASTRepr<T>>),
    Cholesky(Box<crate::ast::ASTRepr<T>>),
}

#[cfg(feature = "linear_algebra")]
#[derive(Debug, Clone, PartialEq)]
pub struct LinearAlgebraCategory<T: NumericType> {
    pub function: LinearAlgebraFunction<T>,
}

#[cfg(feature = "linear_algebra")]
impl<T> LinearAlgebraCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    pub fn to_egglog(&self) -> String {
        match &self.function {
            LinearAlgebraFunction::MatrixMul(a, b) => {
                format!("(LinAlg (MatMulFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::MatrixAdd(a, b) => {
                format!("(LinAlg (MatAddFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::MatrixSub(a, b) => {
                format!("(LinAlg (MatSubFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::MatrixScale(s, m) => format!(
                "(LinAlg (MatScaleFunc {} {}))",
                s.to_egglog(),
                m.to_egglog()
            ),
            LinearAlgebraFunction::Transpose(m) => {
                format!("(LinAlg (TransposeFunc {}))", m.to_egglog())
            }
            LinearAlgebraFunction::Determinant(m) => {
                format!("(LinAlg (DetFunc {}))", m.to_egglog())
            }
            LinearAlgebraFunction::Inverse(m) => format!("(LinAlg (InvFunc {}))", m.to_egglog()),
            LinearAlgebraFunction::Trace(m) => format!("(LinAlg (TraceFunc {}))", m.to_egglog()),
            LinearAlgebraFunction::LeftDivide(a, b) => {
                format!("(LinAlg (LeftDivFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::RightDivide(a, b) => format!(
                "(LinAlg (RightDivFunc {} {}))",
                a.to_egglog(),
                b.to_egglog()
            ),
            LinearAlgebraFunction::DotProduct(a, b) => {
                format!("(LinAlg (DotFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::CrossProduct(a, b) => {
                format!("(LinAlg (CrossFunc {} {}))", a.to_egglog(), b.to_egglog())
            }
            LinearAlgebraFunction::VectorNorm(v) => {
                format!("(LinAlg (NormFunc {}))", v.to_egglog())
            }
            LinearAlgebraFunction::Eigenvalues(m) => {
                format!("(LinAlg (EigValsFunc {}))", m.to_egglog())
            }
            LinearAlgebraFunction::Eigenvectors(m) => {
                format!("(LinAlg (EigVecsFunc {}))", m.to_egglog())
            }
            LinearAlgebraFunction::LU(m) => format!("(LinAlg (LUFunc {}))", m.to_egglog()),
            LinearAlgebraFunction::QR(m) => format!("(LinAlg (QRFunc {}))", m.to_egglog()),
            LinearAlgebraFunction::SVD(m) => format!("(LinAlg (SVDFunc {}))", m.to_egglog()),
            LinearAlgebraFunction::Cholesky(m) => {
                format!("(LinAlg (CholeskyFunc {}))", m.to_egglog())
            }
        }
    }

    pub fn apply_local_rules(
        &self,
        expr: &crate::ast::ASTRepr<T>,
    ) -> Option<crate::ast::ASTRepr<T>> {
        // Linear algebra specific rules would go here
        None
    }
}

// Helper functions for pattern matching
fn extract_sin_squared<T: NumericType + Float>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn extract_cos_squared<T: NumericType + Float>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn extract_sec_squared<T: NumericType>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn extract_tan_squared<T: NumericType>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn extract_cosh_squared<T: NumericType>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn extract_sinh_squared<T: NumericType>(
    _expr: &crate::ast::ASTRepr<T>,
) -> Option<crate::ast::ASTRepr<T>> {
    // Placeholder implementation
    None
}

fn expressions_structurally_equal<T: NumericType>(
    _a: &crate::ast::ASTRepr<T>,
    _b: &crate::ast::ASTRepr<T>,
) -> bool {
    // Placeholder implementation
    false
}

// Trait implementations for easy construction
impl<T: NumericType> TrigCategory<T> {
    pub fn sin(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: TrigFunction::Sin(Box::new(arg)),
        }
    }

    pub fn cos(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: TrigFunction::Cos(Box::new(arg)),
        }
    }

    pub fn tan(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: TrigFunction::Tan(Box::new(arg)),
        }
    }
}

impl<T: NumericType> HyperbolicCategory<T> {
    pub fn sinh(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: HyperbolicFunction::Sinh(Box::new(arg)),
        }
    }

    pub fn cosh(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: HyperbolicFunction::Cosh(Box::new(arg)),
        }
    }

    pub fn tanh(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: HyperbolicFunction::Tanh(Box::new(arg)),
        }
    }
}

#[cfg(feature = "logexp")]
impl<T: NumericType> LogExpCategory<T> {
    pub fn log(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::Log(Box::new(arg)),
        }
    }

    pub fn ln(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::Ln(Box::new(arg)),
        }
    }

    pub fn exp(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::Exp(Box::new(arg)),
        }
    }

    pub fn log_base(base: crate::ast::ASTRepr<T>, arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::LogBase(Box::new(base), Box::new(arg)),
        }
    }

    pub fn log10(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::Log10(Box::new(arg)),
        }
    }

    pub fn log2(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LogExpFunction::Log2(Box::new(arg)),
        }
    }
}

// Convenience constructors for special functions
#[cfg(feature = "special")]
impl<T: NumericType> SpecialCategory<T> {
    pub fn gamma(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: SpecialFunction::Gamma(Box::new(arg)),
        }
    }

    pub fn log_gamma(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: SpecialFunction::LogGamma(Box::new(arg)),
        }
    }

    pub fn beta(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: SpecialFunction::Beta(Box::new(a), Box::new(b)),
        }
    }

    pub fn erf(arg: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: SpecialFunction::Erf(Box::new(arg)),
        }
    }
}

// Convenience constructors for linear algebra operations
#[cfg(feature = "linear_algebra")]
impl<T: NumericType> LinearAlgebraCategory<T> {
    pub fn matrix_mul(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::MatrixMul(Box::new(a), Box::new(b)),
        }
    }

    pub fn matrix_add(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::MatrixAdd(Box::new(a), Box::new(b)),
        }
    }

    pub fn left_divide(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::LeftDivide(Box::new(a), Box::new(b)),
        }
    }

    pub fn right_divide(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::RightDivide(Box::new(a), Box::new(b)),
        }
    }

    pub fn transpose(m: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::Transpose(Box::new(m)),
        }
    }

    pub fn cross_product(a: crate::ast::ASTRepr<T>, b: crate::ast::ASTRepr<T>) -> Self {
        Self {
            function: LinearAlgebraFunction::CrossProduct(Box::new(a), Box::new(b)),
        }
    }
}

impl<T> FunctionCategory<T> for TrigCategory<T>
where
    T: NumericType + Float + std::fmt::Display + std::fmt::Debug + Clone + Default + Send + Sync,
{
    fn to_egglog(&self) -> String {
        match &self.function {
            TrigFunction::Sin(arg) => format!("(Trig (SinFunc {}))", arg.to_egglog()),
            TrigFunction::Cos(arg) => format!("(Trig (CosFunc {}))", arg.to_egglog()),
            TrigFunction::Tan(arg) => format!("(Trig (TanFunc {}))", arg.to_egglog()),
            TrigFunction::Sec(arg) => format!("(Trig (SecFunc {}))", arg.to_egglog()),
            TrigFunction::Csc(arg) => format!("(Trig (CscFunc {}))", arg.to_egglog()),
            TrigFunction::Cot(arg) => format!("(Trig (CotFunc {}))", arg.to_egglog()),
            TrigFunction::Asin(arg) => format!("(Trig (AsinFunc {}))", arg.to_egglog()),
            TrigFunction::Acos(arg) => format!("(Trig (AcosFunc {}))", arg.to_egglog()),
            TrigFunction::Atan(arg) => format!("(Trig (AtanFunc {}))", arg.to_egglog()),
            TrigFunction::Atan2(y, x) => {
                format!("(Trig (Atan2Func {} {}))", y.to_egglog(), x.to_egglog())
            }
        }
    }

    fn apply_local_rules(&self, expr: &crate::ast::ASTRepr<T>) -> Option<crate::ast::ASTRepr<T>> {
        use crate::ast::ASTRepr;

        // Apply trigonometric identities locally
        match expr {
            // sin²(x) + cos²(x) = 1 pattern detection
            ASTRepr::Add(left, right) => {
                if let (Some(sin_arg), Some(cos_arg)) =
                    (extract_sin_squared(left), extract_cos_squared(right))
                {
                    if expressions_structurally_equal(&sin_arg, &cos_arg) {
                        return Some(ASTRepr::Constant(T::from(1.0).unwrap()));
                    }
                }
                None
            }
            // sec²(x) - tan²(x) = 1
            ASTRepr::Sub(left, right) => {
                if let (Some(sec_arg), Some(tan_arg)) =
                    (extract_sec_squared(left), extract_tan_squared(right))
                {
                    if expressions_structurally_equal(&sec_arg, &tan_arg) {
                        return Some(ASTRepr::Constant(T::from(1.0).unwrap()));
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn category_name(&self) -> &'static str {
        "trigonometric"
    }

    fn priority(&self) -> u32 {
        200 // Higher priority for fundamental trig identities
    }
}
