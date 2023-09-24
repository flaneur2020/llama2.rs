#[derive(Debug)]
pub struct Llama2Error {
    pub kind: Llama2ErrorKind,
    pub message: String,
    pub source: Option<Box<dyn std::error::Error>>,
}

impl std::fmt::Display for Llama2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.kind)?;
        write!(f, "{}", self.message)?;
        if let Some(source) = &self.source {
            write!(f, ": {}", source)?;
        }
        Ok(())
    }
}

impl std::error::Error for Llama2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

pub type Result<T> = std::result::Result<T, Llama2Error>;


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Llama2ErrorKind {
    IOError,
    BadInput,
    Unexpected,
    TensorError,
}