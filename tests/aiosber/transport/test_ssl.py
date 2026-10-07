"""Тесты SslContextProvider."""

from __future__ import annotations

import asyncio
import ssl
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.x509.verification import PolicyBuilder, Store, VerificationError

from custom_components.sberhome.aiosber.const import ROOT_CA_PEM, RUSSIAN_TRUSTED_SUB_CA_PEM
from custom_components.sberhome.aiosber.transport import SslContextProvider


async def test_gateway_chain_without_server_sent_intermediate():
    """The bundled CA chain verifies the gateway's leaf-only TLS response offline.

    Public certificate captured on 2026-10-07; no private key or credentials.
    Freeze the validation date so certificate expiry cannot make this flaky.
    """
    context = await SslContextProvider().get()
    leaf = x509.load_pem_x509_certificate(
        (Path(__file__).parent / "fixtures/gateway_missing_intermediate.pem").read_bytes()
    )
    store = Store(
        [x509.load_der_x509_certificate(cert) for cert in context.get_ca_certs(binary_form=True)]
    )
    policy = PolicyBuilder().store(store).time(datetime(2026, 10, 7, tzinfo=UTC))
    chain = policy.build_server_verifier(x509.DNSName("gateway.iot.sberdevices.ru")).verify(
        leaf, []
    )
    assert len(chain) >= 2
    with pytest.raises(VerificationError):
        policy.build_server_verifier(x509.DNSName("unrelated.example")).verify(leaf, [])
    with pytest.raises(VerificationError):
        PolicyBuilder().store(store).time(datetime(2028, 10, 7, tzinfo=UTC)).build_server_verifier(
            x509.DNSName("gateway.iot.sberdevices.ru")
        ).verify(leaf, [])


def test_intermediate_is_signed_by_existing_root():
    """The added intermediate belongs to the already trusted root, not a new CA."""
    root = x509.load_pem_x509_certificate(ROOT_CA_PEM.encode())
    intermediate = x509.load_pem_x509_certificate(RUSSIAN_TRUSTED_SUB_CA_PEM.encode())
    intermediate.verify_directly_issued_by(root)
    assert intermediate.extensions.get_extension_for_class(x509.BasicConstraints).value.ca


async def test_custom_ca_bundle_does_not_add_default_intermediate(monkeypatch):
    """Explicit CA injection keeps its existing semantics without the default bundle."""
    monkeypatch.setattr(
        ssl, "create_default_context", lambda: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    )
    context = await SslContextProvider(ca_pem=ROOT_CA_PEM).get()
    assert context.get_ca_certs(binary_form=True) == [ssl.PEM_cert_to_DER_cert(ROOT_CA_PEM)]


async def test_default_context_preserves_system_trust_and_verification():
    """Adding the CA bundle keeps system trust, hostname checks and strict flags."""
    default = ssl.create_default_context()
    context = await SslContextProvider().get()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.verify_flags == default.verify_flags
    assert set(default.get_ca_certs(binary_form=True)) <= set(
        context.get_ca_certs(binary_form=True)
    )


async def test_get_returns_ssl_context():
    p = SslContextProvider()
    ctx = await p.get()
    assert isinstance(ctx, ssl.SSLContext)


async def test_get_caches_context():
    p = SslContextProvider()
    ctx1 = await p.get()
    ctx2 = await p.get()
    assert ctx1 is ctx2


async def test_concurrent_calls_share_one_build():
    """5 одновременных get() — context создаётся только раз."""
    builds = 0

    class CountingProvider(SslContextProvider):
        def _build(self) -> ssl.SSLContext:
            nonlocal builds
            builds += 1
            return super()._build()

    p = CountingProvider()
    contexts = await asyncio.gather(*[p.get() for _ in range(5)])
    assert all(c is contexts[0] for c in contexts)
    assert builds == 1


async def test_reset_invalidates_cache():
    p = SslContextProvider()
    ctx1 = await p.get()
    p.reset()
    ctx2 = await p.get()
    assert ctx1 is not ctx2


async def test_no_global_state_between_instances():
    """Два разных SslContextProvider имеют разные контексты."""
    p1 = SslContextProvider()
    p2 = SslContextProvider()
    ctx1 = await p1.get()
    ctx2 = await p2.get()
    assert ctx1 is not ctx2


async def test_custom_ca_pem():
    """Можно подать свой CA. Невалидный PEM — должен бросить."""
    p = SslContextProvider(ca_pem="-----BEGIN CERTIFICATE-----\nINVALID\n-----END CERTIFICATE-----")
    with pytest.raises(ssl.SSLError):
        await p.get()
