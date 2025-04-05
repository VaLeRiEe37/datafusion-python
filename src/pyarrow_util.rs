// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Conversions between PyArrow and DataFusion types

use arrow::array::{Array, ArrayData};
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use datafusion::scalar::ScalarValue;
use pyo3::types::{PyAnyMethods, PyList};
use pyo3::{Bound, FromPyObject, PyAny, PyObject, PyResult, Python};

use crate::common::data_type::PyScalarValue;
use crate::errors::PyDataFusionError;

impl FromPyArrow for PyScalarValue {
    fn from_pyarrow_bound(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = value.py();
        let typ = value.getattr("type")?;
        let val = value.call_method0("as_py")?;

        // construct pyarrow array from the python value and pyarrow type
        let factory = py.import("pyarrow")?.getattr("array")?;
        let args = PyList::new(py, [val])?;
        let array = factory.call1((args, typ))?;

        // convert the pyarrow array to rust array using C data interface
        let array = arrow::array::make_array(ArrayData::from_pyarrow_bound(&array)?);
        let scalar = ScalarValue::try_from_array(&array, 0).map_err(PyDataFusionError::from)?;

        Ok(PyScalarValue(scalar))
    }
}

impl<'source> FromPyObject<'source> for PyScalarValue {
    fn extract_bound(value: &Bound<'source, PyAny>) -> PyResult<Self> {
        Self::from_pyarrow_bound(value)
    }
}

pub fn scalar_to_pyarrow(scalar: &ScalarValue, py: Python) -> PyResult<PyObject> {
    let array = scalar.to_array().map_err(PyDataFusionError::from)?;
    // convert to pyarrow array using C data interface
    let pyarray = array.to_data().to_pyarrow(py)?;
    let pyscalar = pyarray.call_method1(py, "__getitem__", (0,))?;

    Ok(pyscalar)
}

/// Convert a DataFusion data type to a pandas-compatible dtype string
pub fn datafusion_to_pandas_dtype(dtype: &DataType) -> String {
    match dtype {
        DataType::Boolean => "bool".to_string(),
        DataType::Int8 => "int8".to_string(),
        DataType::Int16 => "int16".to_string(),
        DataType::Int32 => "int32".to_string(),
        DataType::Int64 => "int64".to_string(),
        DataType::UInt8 => "uint8".to_string(),
        DataType::UInt16 => "uint16".to_string(),
        DataType::UInt32 => "uint32".to_string(),
        DataType::UInt64 => "uint64".to_string(),
        DataType::Float32 => "float32".to_string(),
        DataType::Float64 => "float64".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 => "object".to_string(),
        DataType::Date32 | DataType::Date64 => "datetime64[ns]".to_string(),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => "datetime64[ns]".to_string(),
        DataType::Timestamp(TimeUnit::Microsecond, _) => "datetime64[us]".to_string(),
        DataType::Timestamp(TimeUnit::Millisecond, _) => "datetime64[ms]".to_string(),
        DataType::Timestamp(TimeUnit::Second, _) => "datetime64[s]".to_string(),
        // Add more type mappings in the future if needed
        _ => "object".to_string(), // Default to object for complex types
    }
}

/// Convert a pandas dtype string to a DataFusion data type
pub fn pandas_dtype_to_datafusion(dtype: &str) -> Result<DataType, DataFusionError> {
    match dtype {
        "bool" => Ok(DataType::Boolean),
        "int8" => Ok(DataType::Int8),
        "int16" => Ok(DataType::Int16),
        "int32" | "int" => Ok(DataType::Int32),
        "int64" => Ok(DataType::Int64),
        "uint8" => Ok(DataType::UInt8),
        "uint16" => Ok(DataType::UInt16),
        "uint32" => Ok(DataType::UInt32),
        "uint64" => Ok(DataType::UInt64),
        "float32" | "float" => Ok(DataType::Float32),
        "float64" => Ok(DataType::Float64),
        "object" | "string" => Ok(DataType::Utf8),
        "datetime64[ns]" => Ok(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        "datetime64[us]" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        "datetime64[ms]" => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
        "datetime64[s]" => Ok(DataType::Timestamp(TimeUnit::Second, None)),
        // Add more type mappings in the future if needed
        _ => Err(DataFusionError::Plan(format!("Unsupported pandas dtype: {}", dtype))),
    }
}
