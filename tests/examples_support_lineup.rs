#[path = "../examples/support/mod.rs"]
mod support;

use corinth_canal::{ModelArchitectureClass, ModelFamily, ModelTarget};
use support::{
    cloud_execution_guard, cloud_lineup_path_from_env, load_cloud_lineup, load_safetensors_lineup,
    safetensors_lineup_path_from_env,
};

#[test]
fn cloud_lineup_parses_valid_toml() {
    let tmp = std::env::temp_dir().join(format!(
        "cloud_test_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "test_cloud_model"
family = "olmoe"
cloud_model_id = "test/example-model"
source_url = "https://example.com/model"
target = "cloud"
architecture = "MoE"
active_params = "1B"
total_params = "7B"
provider_format = "nvcf-nim"
required_env_vars = ["TEST_ENDPOINT", "TEST_API_KEY"]
"#,
    )
    .unwrap();

    let entries = load_cloud_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(entries.len(), 1);
    let e = &entries[0];
    assert_eq!(e.slug, "test_cloud_model");
    assert_eq!(e.family, Some(ModelFamily::Olmoe));
    assert_eq!(e.cloud_model_id, "test/example-model");
    assert_eq!(e.target, ModelTarget::Cloud);
    assert_eq!(e.architecture, ModelArchitectureClass::Moe);
    assert_eq!(e.active_params, "1B");
    assert_eq!(e.total_params, "7B");
    assert_eq!(e.provider_format, "nvcf-nim");
    assert_eq!(e.required_env_vars, vec!["TEST_ENDPOINT", "TEST_API_KEY"]);
    assert!(!e.cloud_provider_available());
}

#[test]
fn cloud_execution_guard_fails_when_provider_unavailable() {
    let entry = corinth_canal::CloudModelSpec {
        slug: "test_model".into(),
        family: Some(ModelFamily::Olmoe),
        cloud_model_id: "test/model".into(),
        source_url: "https://example.com".into(),
        target: ModelTarget::Cloud,
        architecture: ModelArchitectureClass::Moe,
        active_params: "1B".into(),
        total_params: "7B".into(),
        provider_format: "nvcf-nim".into(),
        required_env_vars: vec!["UNSET_VAR_XYZ".into()],
    };
    let err = cloud_execution_guard(&entry).unwrap_err();
    assert!(err.contains("UNSET_VAR_XYZ"));
    assert!(err.contains("Dioscuri-Cloud"));
}

#[test]
fn cloud_execution_guard_passes_when_provider_available() {
    let entry = corinth_canal::CloudModelSpec {
        slug: "test_model".into(),
        family: Some(ModelFamily::Olmoe),
        cloud_model_id: "test/model".into(),
        source_url: "https://example.com".into(),
        target: ModelTarget::Cloud,
        architecture: ModelArchitectureClass::Moe,
        active_params: "1B".into(),
        total_params: "7B".into(),
        provider_format: "nvcf-nim".into(),
        required_env_vars: vec![],
    };
    assert!(cloud_execution_guard(&entry).is_ok());
}

#[test]
fn cloud_lineup_unknown_family_reported_on_stderr() {
    let tmp = std::env::temp_dir().join(format!(
        "cloud_unknown_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "unknown_family_model"
family = "totally_fake_family"
cloud_model_id = "fake/model"
source_url = "https://example.com"
target = "cloud"
architecture = "dense"
active_params = "100M"
total_params = "100M"
provider_format = "rest"
"#,
    )
    .unwrap();

    let entries = load_cloud_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(entries.len(), 1);
    assert!(entries[0].family.is_none());
}

#[test]
fn cloud_lineup_path_from_env_returns_none_when_unset() {
    let old = std::env::var_os("CLOUD_LINEUP_CONFIG");
    unsafe {
        std::env::remove_var("CLOUD_LINEUP_CONFIG");
    }
    assert!(cloud_lineup_path_from_env().is_none());
    match old {
        Some(value) => unsafe {
            std::env::set_var("CLOUD_LINEUP_CONFIG", value);
        },
        None => unsafe {
            std::env::remove_var("CLOUD_LINEUP_CONFIG");
        },
    }
}

#[test]
fn safetensors_lineup_parses_valid_toml() {
    let tmp = std::env::temp_dir().join(format!(
        "safetensors_test_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let existing_file = std::env::temp_dir().join(format!(
        "st_dummy_{}.safetensors",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&existing_file, b"test").unwrap();

    std::fs::write(
        &tmp,
        format!(
            r#"
[[model]]
slug = "test_st_model"
family = "olmoe"
path = "{}"
target = "local"
"#,
            existing_file.display()
        ),
    )
    .unwrap();

    let entries = load_safetensors_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&existing_file);

    assert_eq!(entries.len(), 1);
    let e = &entries[0];
    assert_eq!(e.slug, "test_st_model");
    assert_eq!(e.family, Some(ModelFamily::Olmoe));
    assert_eq!(e.target, "local");
}

#[test]
fn safetensors_lineup_skips_missing_paths() {
    let tmp = std::env::temp_dir().join(format!(
        "st_missing_{}.toml",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &tmp,
        r#"
[[model]]
slug = "missing_model"
family = "olmoe"
path = "/nonexistent/path/model.safetensors"
target = "local"
"#,
    )
    .unwrap();

    let entries = load_safetensors_lineup(&tmp).unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert!(entries.is_empty());
}

#[test]
fn safetensors_lineup_path_from_env_returns_none_when_unset() {
    let old = std::env::var_os("SAFETENSORS_LINEUP_CONFIG");
    unsafe {
        std::env::remove_var("SAFETENSORS_LINEUP_CONFIG");
    }
    assert!(safetensors_lineup_path_from_env().is_none());
    match old {
        Some(value) => unsafe {
            std::env::set_var("SAFETENSORS_LINEUP_CONFIG", value);
        },
        None => unsafe {
            std::env::remove_var("SAFETENSORS_LINEUP_CONFIG");
        },
    }
}
