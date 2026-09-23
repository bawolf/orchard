//! In-band secret distribution for Orchard bundles.

use alloc::vec::Vec;
use core::fmt;

use blake2b_simd::{Hash, Params};
use group::ff::PrimeField;
use zcash_note_encryption::{
    BatchDomain, Domain, EphemeralKeyBytes, NotePlaintextBytes, OutPlaintextBytes,
    OutgoingCipherKey, ShieldedOutput, COMPACT_NOTE_SIZE, ENC_CIPHERTEXT_SIZE, NOTE_PLAINTEXT_SIZE,
    OUT_PLAINTEXT_SIZE,
};

use crate::{
    action::Action,
    keys::{
        DiversifiedTransmissionKey, Diversifier, EphemeralPublicKey, EphemeralSecretKey,
        OutgoingViewingKey, PreparedEphemeralPublicKey, PreparedIncomingViewingKey, SharedSecret,
    },
    note::{ExtractedNoteCommitment, NoteVersion, Nullifier, RandomSeed, Rho},
    value::{NoteValue, ValueCommitment},
    Address, Note,
};

const PRF_OCK_ORCHARD_PERSONALIZATION: &[u8; 16] = b"Zcash_Orchardock";

/// Defined in [Zcash Protocol Spec § 5.4.2: Pseudo Random Functions][concreteprfs].
///
/// [concreteprfs]: https://zips.z.cash/protocol/nu5.pdf#concreteprfs
pub(crate) fn prf_ock_orchard(
    ovk: &OutgoingViewingKey,
    cv: &ValueCommitment,
    cmx_bytes: &[u8; 32],
    ephemeral_key: &EphemeralKeyBytes,
) -> OutgoingCipherKey {
    OutgoingCipherKey(
        Params::new()
            .hash_length(32)
            .personal(PRF_OCK_ORCHARD_PERSONALIZATION)
            .to_state()
            .update(ovk.as_ref())
            .update(&cv.to_bytes())
            .update(cmx_bytes)
            .update(ephemeral_key.as_ref())
            .finalize()
            .as_bytes()
            .try_into()
            .unwrap(),
    )
}

fn parse_note_plaintext_without_memo<F>(
    rho: Rho,
    plaintext: &[u8],
    note_version: NoteVersion,
    get_pk_d: F,
) -> Option<(Note, Address)>
where
    F: FnOnce(&Diversifier) -> DiversifiedTransmissionKey,
{
    assert!(plaintext.len() >= COMPACT_NOTE_SIZE);

    // The unwraps below are guaranteed to succeed by the assertion above
    let diversifier = Diversifier::from_bytes(plaintext[1..12].try_into().unwrap());
    let value = NoteValue::from_bytes(plaintext[12..20].try_into().unwrap());
    let rseed = Option::from(RandomSeed::from_bytes(
        plaintext[20..COMPACT_NOTE_SIZE].try_into().unwrap(),
        &rho,
    ))?;

    let pk_d = get_pk_d(&diversifier);

    let recipient = Address::from_parts(diversifier, pk_d);
    let note = Option::from(Note::from_parts(recipient, value, rho, rseed, note_version))?;
    Some((note, recipient))
}

// ============================================================================
// DEDUP LEVER 1: device-local output recovery bound to an already-validated
// `Note`, WITHOUT re-deriving the note commitment (`cmstar`).
//
// The stock `zcash_note_encryption::try_output_recovery_with_*` functions each
// re-derive the note commitment twice per call:
//   1. `parse_note_plaintext_without_memo` -> `Note::from_parts`
//      (constructibility check: one Sinsemilla NoteCommit), and
//   2. `check_note_validity` -> `D::cmstar(note)`
//      (binding check: a second identical Sinsemilla NoteCommit).
// With three recovery calls per action (pkd_esk / ock / ovk) that is six
// redundant NoteCommit evaluations, on top of the one the engine already does
// in `verify_note_commitment`.
//
// During Ironwood bundle verification the engine has ALREADY validated the
// output note against the action's `cmx` (via `verify_note_commitment`, which
// derives `cmx` exactly once). So the only thing output recovery still needs
// to establish is that the ciphertext decrypts to *that same note* and that
// the ZIP-212 ephemeral-key checks hold. We can prove "decrypts to that same
// note" by comparing the decrypted plaintext fields to the already-validated
// `Note` field-by-field, instead of hashing them back into a commitment.
//
// SECURITY EQUIVALENCE: the note commitment is a binding commitment to exactly
// (g_d, pk_d, v, rho, psi(rseed,rho)). `rho` is fixed by the action. The
// diversifier determines `g_d`. Therefore equality of (version, diversifier,
// pk_d, value, rseed) between the decrypted plaintext and the already-validated
// note implies the reconstructed note is byte-identical, hence has the same
// commitment, hence `cmstar(reconstructed) == cmstar(validated) == action.cmx`.
// This is exactly the property `check_note_validity` establishes, so the check
// is NOT weakened — it is the same binding, computed by comparison instead of
// re-hashing. The ZIP-212 `esk`/`epk` checks (scalar-mult / hash-to-curve, not
// Sinsemilla) are preserved verbatim.
// ============================================================================

use chacha20poly1305::{aead::AeadInPlace, ChaCha20Poly1305, KeyInit};
use subtle::ConstantTimeEq;
use zcash_note_encryption::OUT_CIPHERTEXT_SIZE;

/// Decrypts `output`'s note ciphertext with the shared secret derived from
/// (`pk_d`, `esk`), then BINDS the decrypted plaintext to the already-validated
/// `expected` note by field comparison (no `cmstar` recomputation). Returns the
/// recovered memo on success. See the module-level DEDUP LEVER 1 note.
fn recover_bound_inner<P, Output>(
    domain: &NoteEncryptionDomain<P>,
    pk_d: DiversifiedTransmissionKey,
    esk: EphemeralSecretKey,
    output: &Output,
    expected: &Note,
) -> Option<[u8; 512]>
where
    P: DomainVersion,
    Output: ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>,
{
    let ephemeral_key = output.ephemeral_key();
    let shared_secret = <NoteEncryptionDomain<P> as Domain>::ka_agree_enc(&esk, &pk_d);
    let key = <NoteEncryptionDomain<P> as Domain>::kdf(shared_secret, &ephemeral_key);

    let enc_ciphertext = output.enc_ciphertext();
    let mut plaintext = [0u8; NOTE_PLAINTEXT_SIZE];
    plaintext.copy_from_slice(&enc_ciphertext[..NOTE_PLAINTEXT_SIZE]);

    // AEAD authentication failure => reject (identical to the stock path).
    ChaCha20Poly1305::new(key.as_ref().into())
        .decrypt_in_place_detached(
            [0u8; 12][..].into(),
            &[],
            &mut plaintext,
            enc_ciphertext[NOTE_PLAINTEXT_SIZE..].into(),
        )
        .ok()?;

    // --- Bind the decrypted plaintext to the already-validated note. ---
    //
    // MUST-FIX #1 (Fable review): the binding must not rest on the caller having
    // built `expected` from the *same* action. `rho` is the one NoteCommit input
    // that never appears in the plaintext (upstream takes it from `domain.rho`),
    // and the accepted note version is a property of the *domain*, not of
    // `expected`. So both are now enforced against `domain` inside this function,
    // exactly as `zcash_note_encryption` does, at zero extra Sinsemilla cost.
    //
    // (rho) the domain's rho (= Rho::from_nf_old(action.spend().nullifier)) must
    //       match the validated note's rho. Together with the field checks below
    //       this pins every free input of NoteCommit to the action's `cmx`.
    if domain.rho != expected.rho() {
        return None;
    }
    // (a) plaintext version lead byte, filtered through the *domain* policy
    //     (upstream's `domain.policy.note_version(plaintext)`), then required to
    //     equal the validated note's version.
    let plaintext_version = domain.policy.note_version(&plaintext)?;
    if plaintext_version != expected.version() {
        return None;
    }
    // (b) diversifier (determines g_d).
    let diversifier = Diversifier::from_bytes(plaintext[1..12].try_into().unwrap());
    if diversifier != expected.recipient().diversifier() {
        return None;
    }
    // (c) value.
    let value = NoteValue::from_bytes(plaintext[12..20].try_into().unwrap());
    if value != expected.value() {
        return None;
    }
    // (d) rseed (determines psi and rcm).
    if plaintext[20..COMPACT_NOTE_SIZE] != expected.rseed().as_bytes()[..] {
        return None;
    }
    // (e) transmission key binding: the pk_d that keyed decryption (for the
    //     pkd_esk path) or that was recovered from the outgoing ciphertext (for
    //     the ock/ovk paths) must match the validated recipient.
    if &pk_d != expected.recipient().pk_d() {
        return None;
    }

    // --- ZIP-212 ephemeral-key checks (preserved verbatim; no Sinsemilla). ---
    let derived_esk = <NoteEncryptionDomain<P> as Domain>::derive_esk(expected)?;
    if bool::from(!derived_esk.ct_eq(&esk)) {
        return None;
    }
    if !bool::from(
        <NoteEncryptionDomain<P> as Domain>::epk_bytes(
            &<NoteEncryptionDomain<P> as Domain>::ka_derive_public(expected, &derived_esk),
        )
        .ct_eq(&ephemeral_key),
    ) {
        return None;
    }

    let memo: [u8; 512] = plaintext[COMPACT_NOTE_SIZE..NOTE_PLAINTEXT_SIZE]
        .try_into()
        .unwrap();
    Some(memo)
}

/// Device-local equivalent of `try_output_recovery_with_pkd_esk`, bound to an
/// already-validated note (no `cmstar` recomputation). Returns the memo.
///
/// Takes the concrete [`NoteEncryptionDomain`] (not a generic `Domain`) so that
/// the note's `rho` and accepted version are checked against the *domain* (i.e.
/// the action), per MUST-FIX #1.
pub fn recover_output_bound_with_pkd_esk<P, Output>(
    domain: &NoteEncryptionDomain<P>,
    pk_d: DiversifiedTransmissionKey,
    esk: EphemeralSecretKey,
    output: &Output,
    expected: &Note,
) -> Option<[u8; 512]>
where
    P: DomainVersion,
    Output: ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>,
{
    recover_bound_inner::<P, Output>(domain, pk_d, esk, output, expected)
}

/// Device-local equivalent of `try_output_recovery_with_ock`, bound to an
/// already-validated note (no `cmstar` recomputation). Returns the memo.
pub fn recover_output_bound_with_ock<P, Output>(
    domain: &NoteEncryptionDomain<P>,
    ock: &OutgoingCipherKey,
    output: &Output,
    out_ciphertext: &[u8; OUT_CIPHERTEXT_SIZE],
    expected: &Note,
) -> Option<[u8; 512]>
where
    P: DomainVersion,
    Output: ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>,
{
    let mut op = OutPlaintextBytes([0; OUT_PLAINTEXT_SIZE]);
    op.0.copy_from_slice(&out_ciphertext[..OUT_PLAINTEXT_SIZE]);

    ChaCha20Poly1305::new(ock.as_ref().into())
        .decrypt_in_place_detached(
            [0u8; 12][..].into(),
            &[],
            &mut op.0,
            out_ciphertext[OUT_PLAINTEXT_SIZE..].into(),
        )
        .ok()?;

    let pk_d = <NoteEncryptionDomain<P> as Domain>::extract_pk_d(&op)?;
    let esk = <NoteEncryptionDomain<P> as Domain>::extract_esk(&op)?;

    recover_bound_inner::<P, Output>(domain, pk_d, esk, output, expected)
}

/// Device-local equivalent of `try_output_recovery_with_ovk`, bound to an
/// already-validated note (no `cmstar` recomputation). Returns the memo.
pub fn recover_output_bound_with_ovk<P, Output>(
    domain: &NoteEncryptionDomain<P>,
    ovk: &OutgoingViewingKey,
    output: &Output,
    cv: &ValueCommitment,
    out_ciphertext: &[u8; OUT_CIPHERTEXT_SIZE],
    expected: &Note,
) -> Option<[u8; 512]>
where
    P: DomainVersion,
    Output: ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>,
{
    let ock = <NoteEncryptionDomain<P> as Domain>::derive_ock(
        ovk,
        cv,
        &output.cmstar_bytes(),
        &output.ephemeral_key(),
    );
    recover_output_bound_with_ock::<P, Output>(domain, &ock, output, out_ciphertext, expected)
}

mod sealed {
    /// Marker trait that prevents external `DomainVersion` implementations.
    pub trait Sealed {}
}

trait DomainPolicy {
    fn note_version(&self, plaintext: &[u8]) -> Option<NoteVersion>;
}

/// A sealed marker trait for note encryption domains with a fixed note plaintext version.
///
/// This trait is sealed so that only this crate can define supported note encryption
/// domains.
pub trait DomainVersion: sealed::Sealed + Default {
    /// The note plaintext version accepted by this domain during parsing and decryption.
    const NOTE_VERSION: NoteVersion;
}

impl<V: DomainVersion> DomainPolicy for V {
    fn note_version(&self, plaintext: &[u8]) -> Option<NoteVersion> {
        if plaintext.first().copied() == Some(V::NOTE_VERSION.lead_byte()) {
            Some(V::NOTE_VERSION)
        } else {
            None
        }
    }
}

/// Marker type for Orchard note encryption domains.
#[derive(Default, Debug)]
pub struct OrchardVersion;

impl sealed::Sealed for OrchardVersion {}

impl DomainVersion for OrchardVersion {
    const NOTE_VERSION: NoteVersion = NoteVersion::V2;
}

/// Marker type for Ironwood note encryption domains.
#[derive(Default, Debug)]
pub struct IronwoodVersion;

impl sealed::Sealed for IronwoodVersion {}

impl DomainVersion for IronwoodVersion {
    const NOTE_VERSION: NoteVersion = NoteVersion::V3;
}

#[derive(Debug)]
pub(crate) struct BundleDomainPolicy {
    note_version: NoteVersion,
}

impl DomainPolicy for BundleDomainPolicy {
    fn note_version(&self, plaintext: &[u8]) -> Option<NoteVersion> {
        let note_version = NoteVersion::from_lead_byte(*plaintext.first()?)?;
        if note_version == self.note_version {
            Some(note_version)
        } else {
            None
        }
    }
}

/// Note encryption logic for a note plaintext version policy.
///
/// The policy type `P` selects which note plaintext version is accepted during
/// parsing and decryption. Encryption uses the version recorded by the note.
#[derive(Debug)]
pub struct NoteEncryptionDomain<P> {
    rho: Rho,
    policy: P,
}

impl<P> memuse::DynamicUsage for NoteEncryptionDomain<P> {
    fn dynamic_usage(&self) -> usize {
        self.rho.dynamic_usage()
    }

    fn dynamic_usage_bounds(&self) -> (usize, Option<usize>) {
        self.rho.dynamic_usage_bounds()
    }
}

impl<V: DomainVersion> NoteEncryptionDomain<V> {
    pub(crate) fn from_rho(rho: Rho) -> Self {
        Self {
            rho,
            policy: V::default(),
        }
    }

    /// Constructs a domain that can be used to trial-decrypt this action's output note.
    pub fn for_action<T>(act: &Action<T>) -> Self {
        Self::from_rho(act.rho())
    }

    /// Constructs a domain that can be used to trial-decrypt a PCZT action's output note.
    pub fn for_pczt_action(act: &crate::pczt::Action) -> Self {
        Self::from_rho(Rho::from_nf_old(act.spend().nullifier))
    }

    /// Constructs a domain that can be used to trial-decrypt this compact action's output note.
    pub fn for_compact_action(act: &CompactAction) -> Self {
        Self::from_rho(act.rho())
    }
}

/// Orchard-specific note encryption logic.
///
/// This domain accepts only [`NoteVersion::V2`] note plaintexts, which use lead
/// byte `0x02`.
pub type OrchardDomain = NoteEncryptionDomain<OrchardVersion>;

/// Ironwood-specific note encryption logic.
///
/// This domain is otherwise identical to [`OrchardDomain`], but accepts only
/// [`NoteVersion::V3`] note plaintexts, which use lead byte `0x03`.
pub type IronwoodDomain = NoteEncryptionDomain<IronwoodVersion>;

/// Note encryption logic restricted to a single note plaintext version.
///
/// This domain is used by public bundle helpers that are given the bundle's
/// [`NoteVersion`]. Trial decryption still happens once; after decryption
/// succeeds, the revealed note plaintext lead byte selects the note version, which is
/// enforced to match the expected one.
pub(crate) type BundleDomain = NoteEncryptionDomain<BundleDomainPolicy>;

impl BundleDomain {
    /// Constructs a domain that can be used to trial-decrypt this action's
    /// output note as a note of `note_version`.
    pub(crate) fn for_action<T>(act: &Action<T>, note_version: NoteVersion) -> Self {
        Self {
            rho: act.rho(),
            policy: BundleDomainPolicy { note_version },
        }
    }
}

impl<P: DomainPolicy> Domain for NoteEncryptionDomain<P> {
    type EphemeralSecretKey = EphemeralSecretKey;
    type EphemeralPublicKey = EphemeralPublicKey;
    type PreparedEphemeralPublicKey = PreparedEphemeralPublicKey;
    type SharedSecret = SharedSecret;
    type SymmetricKey = Hash;
    type Note = Note;
    type Recipient = Address;
    type DiversifiedTransmissionKey = DiversifiedTransmissionKey;
    type IncomingViewingKey = PreparedIncomingViewingKey;
    type OutgoingViewingKey = OutgoingViewingKey;
    type ValueCommitment = ValueCommitment;
    type ExtractedCommitment = ExtractedNoteCommitment;
    type ExtractedCommitmentBytes = [u8; 32];
    type Memo = [u8; 512]; // TODO use a more interesting type

    fn derive_esk(note: &Self::Note) -> Option<Self::EphemeralSecretKey> {
        Some(note.esk())
    }

    fn get_pk_d(note: &Self::Note) -> Self::DiversifiedTransmissionKey {
        *note.recipient().pk_d()
    }

    fn prepare_epk(epk: Self::EphemeralPublicKey) -> Self::PreparedEphemeralPublicKey {
        PreparedEphemeralPublicKey::new(epk)
    }

    fn ka_derive_public(
        note: &Self::Note,
        esk: &Self::EphemeralSecretKey,
    ) -> Self::EphemeralPublicKey {
        esk.derive_public(note.recipient().g_d())
    }

    fn ka_agree_enc(
        esk: &Self::EphemeralSecretKey,
        pk_d: &Self::DiversifiedTransmissionKey,
    ) -> Self::SharedSecret {
        esk.agree(pk_d)
    }

    fn ka_agree_dec(
        ivk: &Self::IncomingViewingKey,
        epk: &Self::PreparedEphemeralPublicKey,
    ) -> Self::SharedSecret {
        epk.agree(ivk)
    }

    fn kdf(secret: Self::SharedSecret, ephemeral_key: &EphemeralKeyBytes) -> Self::SymmetricKey {
        secret.kdf_orchard(ephemeral_key)
    }

    fn note_plaintext_bytes(note: &Self::Note, memo: &Self::Memo) -> NotePlaintextBytes {
        let mut np = [0; NOTE_PLAINTEXT_SIZE];
        np[0] = note.version().lead_byte();
        np[1..12].copy_from_slice(note.recipient().diversifier().as_array());
        np[12..20].copy_from_slice(&note.value().to_bytes());
        np[20..52].copy_from_slice(note.rseed().as_bytes());
        np[52..].copy_from_slice(memo);
        NotePlaintextBytes(np)
    }

    fn derive_ock(
        ovk: &Self::OutgoingViewingKey,
        cv: &Self::ValueCommitment,
        cmstar_bytes: &Self::ExtractedCommitmentBytes,
        ephemeral_key: &EphemeralKeyBytes,
    ) -> OutgoingCipherKey {
        prf_ock_orchard(ovk, cv, cmstar_bytes, ephemeral_key)
    }

    fn outgoing_plaintext_bytes(
        note: &Self::Note,
        esk: &Self::EphemeralSecretKey,
    ) -> OutPlaintextBytes {
        let mut op = [0; OUT_PLAINTEXT_SIZE];
        op[..32].copy_from_slice(&note.recipient().pk_d().to_bytes());
        op[32..].copy_from_slice(&esk.0.to_repr());
        OutPlaintextBytes(op)
    }

    fn epk_bytes(epk: &Self::EphemeralPublicKey) -> EphemeralKeyBytes {
        epk.to_bytes()
    }

    fn epk(ephemeral_key: &EphemeralKeyBytes) -> Option<Self::EphemeralPublicKey> {
        EphemeralPublicKey::from_bytes(&ephemeral_key.0).into()
    }

    fn cmstar(note: &Self::Note) -> Self::ExtractedCommitment {
        note.commitment().into()
    }

    fn parse_note_plaintext_without_memo_ivk(
        &self,
        ivk: &Self::IncomingViewingKey,
        plaintext: &[u8],
    ) -> Option<(Self::Note, Self::Recipient)> {
        let note_version = self.policy.note_version(plaintext)?;
        parse_note_plaintext_without_memo(self.rho, plaintext, note_version, |diversifier| {
            DiversifiedTransmissionKey::derive(ivk, diversifier)
        })
    }

    fn parse_note_plaintext_without_memo_ovk(
        &self,
        pk_d: &Self::DiversifiedTransmissionKey,
        plaintext: &NotePlaintextBytes,
    ) -> Option<(Self::Note, Self::Recipient)> {
        let note_version = self.policy.note_version(&plaintext.0)?;
        parse_note_plaintext_without_memo(self.rho, &plaintext.0, note_version, |_| *pk_d)
    }

    fn extract_memo(&self, plaintext: &NotePlaintextBytes) -> Self::Memo {
        plaintext.0[COMPACT_NOTE_SIZE..NOTE_PLAINTEXT_SIZE]
            .try_into()
            .unwrap()
    }

    fn extract_pk_d(out_plaintext: &OutPlaintextBytes) -> Option<Self::DiversifiedTransmissionKey> {
        DiversifiedTransmissionKey::from_bytes(out_plaintext.0[0..32].try_into().unwrap()).into()
    }

    fn extract_esk(out_plaintext: &OutPlaintextBytes) -> Option<Self::EphemeralSecretKey> {
        EphemeralSecretKey::from_bytes(out_plaintext.0[32..OUT_PLAINTEXT_SIZE].try_into().unwrap())
            .into()
    }
}

impl<P: DomainPolicy> BatchDomain for NoteEncryptionDomain<P> {
    fn batch_kdf<'a>(
        items: impl Iterator<Item = (Option<Self::SharedSecret>, &'a EphemeralKeyBytes)>,
    ) -> Vec<Option<Self::SymmetricKey>> {
        batch_kdf(items)
    }

    fn batch_epk(
        ephemeral_keys: impl Iterator<Item = EphemeralKeyBytes>,
    ) -> Vec<(Option<Self::PreparedEphemeralPublicKey>, EphemeralKeyBytes)> {
        // Prepare the whole batch with GLV windows, sharing one batch
        // normalization across every key (a single field inversion, where
        // per-item preparation pays one per key).
        let (epks, ephemeral_keys): (Vec<_>, Vec<_>) = ephemeral_keys
            .map(|ephemeral_key| (Self::epk(&ephemeral_key), ephemeral_key))
            .unzip();
        PreparedEphemeralPublicKey::batch_tabled(epks)
            .into_iter()
            .zip(ephemeral_keys)
            .collect()
    }

    fn batch_ka_agree_dec<'a>(
        ivk: &Self::IncomingViewingKey,
        epks: impl Iterator<Item = Option<&'a Self::PreparedEphemeralPublicKey>>,
    ) -> Vec<Option<Self::SharedSecret>>
    where
        Self::PreparedEphemeralPublicKey: 'a,
    {
        // One GLV decomposition and digit recoding of the viewing key for the
        // whole batch; each ephemeral key's window is then consumed by a
        // shared-doubling ladder.
        let decomposed =
            pasta_curves::glv::Decomposed::<pasta_curves::pallas::Point>::new(&ivk.raw_scalar());
        epks.map(|epk| epk.map(|epk| epk.agree_with(ivk, &decomposed)))
            .collect()
    }
}

fn batch_kdf<'a>(
    items: impl Iterator<Item = (Option<SharedSecret>, &'a EphemeralKeyBytes)>,
) -> Vec<Option<Hash>> {
    let (shared_secrets, ephemeral_keys): (Vec<_>, Vec<_>) = items.unzip();

    SharedSecret::batch_to_affine(shared_secrets)
        .zip(ephemeral_keys)
        .map(|(secret, ephemeral_key)| {
            secret.map(|dhsecret| SharedSecret::kdf_orchard_inner(dhsecret, ephemeral_key))
        })
        .collect()
}

impl<P: DomainPolicy, T> ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>
    for Action<T>
{
    fn ephemeral_key(&self) -> EphemeralKeyBytes {
        EphemeralKeyBytes(self.encrypted_note().epk_bytes)
    }

    fn cmstar_bytes(&self) -> [u8; 32] {
        self.cmx().to_bytes()
    }

    fn enc_ciphertext(&self) -> &[u8; ENC_CIPHERTEXT_SIZE] {
        &self.encrypted_note().enc_ciphertext
    }
}

impl<P: DomainPolicy> ShieldedOutput<NoteEncryptionDomain<P>, ENC_CIPHERTEXT_SIZE>
    for crate::pczt::Action
{
    fn ephemeral_key(&self) -> EphemeralKeyBytes {
        EphemeralKeyBytes(self.output().encrypted_note().epk_bytes)
    }

    fn cmstar_bytes(&self) -> [u8; 32] {
        self.output().cmx().to_bytes()
    }

    fn enc_ciphertext(&self) -> &[u8; ENC_CIPHERTEXT_SIZE] {
        &self.output().encrypted_note().enc_ciphertext
    }
}

impl<P: DomainPolicy> ShieldedOutput<NoteEncryptionDomain<P>, COMPACT_NOTE_SIZE> for CompactAction {
    fn ephemeral_key(&self) -> EphemeralKeyBytes {
        EphemeralKeyBytes(self.ephemeral_key.0)
    }

    fn cmstar_bytes(&self) -> [u8; 32] {
        self.cmx.to_bytes()
    }

    fn enc_ciphertext(&self) -> &[u8; COMPACT_NOTE_SIZE] {
        &self.enc_ciphertext
    }
}

/// Implementation of in-band secret distribution for Orchard bundles.
///
/// This is the [`NoteEncryption`] instantiation for [`OrchardDomain`]. Encryption
/// behavior is shared with [`IronwoodNoteEncryption`]: the note plaintext lead
/// byte is selected from [`crate::Note::version`], while the domain type
/// controls which note plaintext versions are accepted during parsing and
/// decryption.
///
/// [`NoteEncryption`]: zcash_note_encryption::NoteEncryption
pub type OrchardNoteEncryption = zcash_note_encryption::NoteEncryption<OrchardDomain>;
/// Implementation of in-band secret distribution for Ironwood bundles.
///
/// This is the [`NoteEncryption`] instantiation for [`IronwoodDomain`]. Encryption
/// behavior is shared with [`OrchardNoteEncryption`]: the note plaintext lead
/// byte is selected from [`crate::Note::version`], while the domain type
/// controls which note plaintext versions are accepted during parsing and
/// decryption.
///
/// [`NoteEncryption`]: zcash_note_encryption::NoteEncryption
pub type IronwoodNoteEncryption = zcash_note_encryption::NoteEncryption<IronwoodDomain>;

/// A compact Action for light clients.
#[derive(Clone)]
pub struct CompactAction {
    nullifier: Nullifier,
    cmx: ExtractedNoteCommitment,
    ephemeral_key: EphemeralKeyBytes,
    enc_ciphertext: [u8; 52],
}

impl fmt::Debug for CompactAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CompactAction")
    }
}

impl<T> From<&Action<T>> for CompactAction {
    fn from(action: &Action<T>) -> Self {
        CompactAction {
            nullifier: *action.nullifier(),
            cmx: *action.cmx(),
            ephemeral_key: EphemeralKeyBytes(action.encrypted_note().epk_bytes),
            enc_ciphertext: action.encrypted_note().enc_ciphertext[..52]
                .try_into()
                .unwrap(),
        }
    }
}

impl CompactAction {
    /// Create a CompactAction from its constituent parts
    pub fn from_parts(
        nullifier: Nullifier,
        cmx: ExtractedNoteCommitment,
        ephemeral_key: EphemeralKeyBytes,
        enc_ciphertext: [u8; 52],
    ) -> Self {
        Self {
            nullifier,
            cmx,
            ephemeral_key,
            enc_ciphertext,
        }
    }

    /// Returns the nullifier of the note being spent.
    pub fn nullifier(&self) -> Nullifier {
        self.nullifier
    }

    /// Returns the commitment to the new note being created.
    pub fn cmx(&self) -> ExtractedNoteCommitment {
        self.cmx
    }

    /// Obtains the [`Rho`] value that was used to construct the new note being created.
    pub fn rho(&self) -> Rho {
        Rho::from_nf_old(self.nullifier)
    }
}

/// Utilities for constructing test data.
#[cfg(feature = "test-dependencies")]
pub mod testing {
    use rand::RngCore;
    use zcash_note_encryption::Domain;

    use crate::{
        keys::OutgoingViewingKey,
        note::{ExtractedNoteCommitment, NoteVersion, Nullifier, RandomSeed, Rho},
        value::NoteValue,
        Address, Note,
    };

    use super::{CompactAction, OrchardDomain, OrchardNoteEncryption};

    /// Creates a fake `CompactAction` paying the given recipient the specified value.
    ///
    /// Returns the `CompactAction` and the new note.
    pub fn fake_compact_action<R: RngCore>(
        rng: &mut R,
        nf_old: Nullifier,
        recipient: Address,
        value: NoteValue,
        ovk: Option<OutgoingViewingKey>,
    ) -> (CompactAction, Note) {
        let rho = Rho::from_nf_old(nf_old);
        let rseed = {
            loop {
                let mut bytes = [0; 32];
                rng.fill_bytes(&mut bytes);
                let rseed = RandomSeed::from_bytes(bytes, &rho);
                if rseed.is_some().into() {
                    break rseed.unwrap();
                }
            }
        };
        let note = Note::from_parts(recipient, value, rho, rseed, NoteVersion::V2).unwrap();
        let encryptor = OrchardNoteEncryption::new(ovk, note, [0u8; 512]);
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        let ephemeral_key = OrchardDomain::epk_bytes(encryptor.epk());
        let enc_ciphertext = encryptor.encrypt_note_plaintext();

        (
            CompactAction {
                nullifier: nf_old,
                cmx,
                ephemeral_key,
                enc_ciphertext: enc_ciphertext.as_ref()[..52].try_into().unwrap(),
            },
            note,
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use rand::rngs::OsRng;
    use zcash_note_encryption::{
        batch, try_compact_note_decryption, try_note_decryption, try_output_recovery_with_ovk,
        BatchDomain, Domain, EphemeralKeyBytes, NoteEncryption,
    };

    use super::{
        prf_ock_orchard, CompactAction, DomainVersion, IronwoodDomain, IronwoodNoteEncryption,
        IronwoodVersion, NoteEncryptionDomain, OrchardDomain, OrchardNoteEncryption,
        OrchardVersion,
    };
    use crate::{
        action::Action,
        keys::{
            DiversifiedTransmissionKey, Diversifier, EphemeralSecretKey, FullViewingKey,
            IncomingViewingKey, OutgoingViewingKey, PreparedIncomingViewingKey, Scope, SpendingKey,
        },
        note::{
            ExtractedNoteCommitment, NoteVersion, Nullifier, RandomSeed, Rho,
            TransmittedNoteCiphertext,
        },
        primitives::redpallas,
        value::{NoteValue, ValueCommitTrapdoor, ValueCommitment, ValueSum},
        Address, Note,
    };

    fn v3_encrypted_action() -> (
        Action<()>,
        PreparedIncomingViewingKey,
        Note,
        Address,
        [u8; 512],
    ) {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk = crate::keys::FullViewingKey::from(&sk);
        let incoming_viewing_key = fvk.to_ivk(Scope::External);
        let prepared_ivk = PreparedIncomingViewingKey::new(&incoming_viewing_key);
        let recipient = fvk.address_at(0u32, Scope::External);
        let nf_old = Nullifier::dummy(&mut rng);
        let rho = Rho::from_nf_old(nf_old);
        let note = Note::new(
            recipient,
            NoteValue::from_raw(5),
            rho,
            NoteVersion::V3,
            &mut rng,
        );
        let memo = [7u8; 512];
        let cv_net = ValueCommitment::derive(ValueSum::from_raw(5), ValueCommitTrapdoor::zero());
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        let encryptor = IronwoodNoteEncryption::new(Some(fvk.to_ovk(Scope::External)), note, memo);
        let encrypted_note = TransmittedNoteCiphertext {
            epk_bytes: IronwoodDomain::epk_bytes(encryptor.epk()).0,
            enc_ciphertext: encryptor.encrypt_note_plaintext(),
            out_ciphertext: encryptor.encrypt_outgoing_plaintext(&cv_net, &cmx, &mut rng),
        };
        let action = Action::from_parts(
            nf_old,
            redpallas::VerificationKey::dummy(),
            cmx,
            encrypted_note,
            cv_net,
            (),
        )
        .expect("a dummy verification key is unlikely to be the identity");

        (action, prepared_ivk, note, recipient, memo)
    }

    #[test]
    fn test_vectors() {
        let test_vectors = crate::test_vectors::note_encryption::test_vectors();

        for tv in test_vectors {
            //
            // Load the test vector components
            //

            // Recipient key material
            let ivk = PreparedIncomingViewingKey::new(
                &IncomingViewingKey::from_bytes(&tv.incoming_viewing_key).unwrap(),
            );
            let ovk = OutgoingViewingKey::from(tv.ovk);
            let d = Diversifier::from_bytes(tv.default_d);
            let pk_d = DiversifiedTransmissionKey::from_bytes(&tv.default_pk_d).unwrap();

            // Received Action
            let cv_net = ValueCommitment::from_bytes(&tv.cv_net).unwrap();
            let nf_old = Nullifier::from_bytes(&tv.nf_old).unwrap();
            let rho = Rho::from_nf_old(nf_old);
            let cmx = ExtractedNoteCommitment::from_bytes(&tv.cmx).unwrap();

            let esk = EphemeralSecretKey::from_bytes(&tv.esk).unwrap();
            let ephemeral_key = EphemeralKeyBytes(tv.ephemeral_key);

            // Details about the expected note
            let value = NoteValue::from_raw(tv.v);
            let rseed = RandomSeed::from_bytes(tv.rseed, &rho).unwrap();

            //
            // Test the individual components
            //

            let shared_secret = esk.agree(&pk_d);
            assert_eq!(shared_secret.to_bytes(), tv.shared_secret);

            let k_enc = shared_secret.kdf_orchard(&ephemeral_key);
            assert_eq!(k_enc.as_bytes(), tv.k_enc);

            let ock = prf_ock_orchard(&ovk, &cv_net, &cmx.to_bytes(), &ephemeral_key);
            assert_eq!(ock.as_ref(), tv.ock);

            let recipient = Address::from_parts(d, pk_d);
            let note_version = NoteVersion::V2;
            let note = Note::from_parts(recipient, value, rho, rseed, note_version).unwrap();
            assert_eq!(ExtractedNoteCommitment::from(note.commitment()), cmx);

            let action = Action::from_parts(
                // nf_old is the nullifier revealed by the receiving Action.
                nf_old,
                // We don't need a real rk for this test.
                redpallas::VerificationKey::dummy(),
                cmx,
                TransmittedNoteCiphertext {
                    epk_bytes: ephemeral_key.0,
                    enc_ciphertext: tv.c_enc,
                    out_ciphertext: tv.c_out,
                },
                cv_net.clone(),
                (),
            )
            .expect("a key returned by VerificationKey::dummy() is vanishingly unlikely to be the identity");

            //
            // Test decryption
            // (Tested first because it only requires immutable references.)
            //

            let domain = OrchardDomain::from_rho(rho);

            match try_note_decryption(&domain, &ivk, &action) {
                Some((decrypted_note, decrypted_to, decrypted_memo)) => {
                    assert_eq!(decrypted_note, note);
                    assert_eq!(decrypted_to, recipient);
                    assert_eq!(&decrypted_memo[..], &tv.memo[..]);
                }
                None => panic!("Note decryption failed"),
            }

            match try_compact_note_decryption(&domain, &ivk, &CompactAction::from(&action)) {
                Some((decrypted_note, decrypted_to)) => {
                    assert_eq!(decrypted_note, note);
                    assert_eq!(decrypted_to, recipient);
                }
                None => panic!("Compact note decryption failed"),
            }

            match try_output_recovery_with_ovk(&domain, &ovk, &action, &cv_net, &tv.c_out) {
                Some((decrypted_note, decrypted_to, decrypted_memo)) => {
                    assert_eq!(decrypted_note, note);
                    assert_eq!(decrypted_to, recipient);
                    assert_eq!(&decrypted_memo[..], &tv.memo[..]);
                }
                None => panic!("Output recovery failed"),
            }

            //
            // Test encryption
            //

            let ne = OrchardNoteEncryption::new_with_esk(esk, Some(ovk), note, tv.memo);

            assert_eq!(ne.encrypt_note_plaintext().as_ref(), &tv.c_enc[..]);
            assert_eq!(
                &ne.encrypt_outgoing_plaintext(&cv_net, &cmx, &mut OsRng)[..],
                &tv.c_out[..]
            );
        }
    }

    #[test]
    fn domains_accept_only_their_note_plaintext_versions() {
        let mut rng = OsRng;
        let sk = crate::keys::SpendingKey::random(&mut rng);
        let fvk = crate::keys::FullViewingKey::from(&sk);
        let recipient = fvk.address_at(0u32, crate::keys::Scope::External);
        let rho = Rho::from_nf_old(Nullifier::dummy(&mut rng));
        let memo = [0u8; 512];

        let note_v2 = Note::new(
            recipient,
            NoteValue::from_raw(5),
            rho,
            NoteVersion::V2,
            &mut rng,
        );
        let note_v3 = Note::new(
            recipient,
            NoteValue::from_raw(5),
            rho,
            NoteVersion::V3,
            &mut rng,
        );
        let orchard_domain = OrchardDomain::from_rho(rho);
        let ironwood_domain = IronwoodDomain::from_rho(rho);

        let np_v2 = OrchardDomain::note_plaintext_bytes(&note_v2, &memo);
        let np_v3 = IronwoodDomain::note_plaintext_bytes(&note_v3, &memo);
        let pk_d = recipient.pk_d();

        assert_eq!(
            orchard_domain
                .parse_note_plaintext_without_memo_ovk(pk_d, &np_v2)
                .map(|(note, _)| note),
            Some(note_v2)
        );
        assert_eq!(
            ironwood_domain
                .parse_note_plaintext_without_memo_ovk(pk_d, &np_v3)
                .map(|(note, _)| note),
            Some(note_v3)
        );
        assert!(orchard_domain
            .parse_note_plaintext_without_memo_ovk(pk_d, &np_v3)
            .is_none());
        assert!(ironwood_domain
            .parse_note_plaintext_without_memo_ovk(pk_d, &np_v2)
            .is_none());
    }

    #[test]
    fn ironwood_domain_decrypts_v3_encrypted_outputs() {
        let (action, ivk, note, recipient, memo) = v3_encrypted_action();
        let domain = IronwoodDomain::for_action(&action);

        assert_eq!(
            try_note_decryption(&domain, &ivk, &action),
            Some((note, recipient, memo))
        );
    }

    #[test]
    fn orchard_domain_rejects_v3_encrypted_outputs() {
        let (action, ivk, _, _, _) = v3_encrypted_action();
        let domain = OrchardDomain::for_action(&action);

        assert!(try_note_decryption(&domain, &ivk, &action).is_none());
    }

    #[test]
    fn ironwood_domain_decrypts_v3_compact_outputs() {
        let (action, ivk, note, recipient, _) = v3_encrypted_action();
        let domain = IronwoodDomain::for_action(&action);
        let compact = CompactAction::from(&action);

        assert_eq!(
            try_compact_note_decryption(&domain, &ivk, &compact),
            Some((note, recipient))
        );
    }

    // ---------------------------------------------------------------------
    // DEDUP PROOF: builds a full Ironwood V3 action retaining every field the
    // engine needs, so we can model both the stock and the deduped per-action
    // verification paths and count Sinsemilla `hash_to_point` evaluations.
    // ---------------------------------------------------------------------
    struct FullAction {
        fvk: crate::keys::FullViewingKey,
        action: Action<()>,
        note: Note,
        recipient: Address,
        memo: [u8; 512],
        ovk: OutgoingViewingKey,
        cv_net: ValueCommitment,
        out_ct: [u8; 80],
        epk: EphemeralKeyBytes,
        ock: super::OutgoingCipherKey,
    }

    fn full_v3_action() -> FullAction {
        use crate::keys::FullViewingKey;
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk = FullViewingKey::from(&sk);
        let ovk = fvk.to_ovk(Scope::External);
        let recipient = fvk.address_at(0u32, Scope::External);
        let nf_old = Nullifier::dummy(&mut rng);
        let rho = Rho::from_nf_old(nf_old);
        let note = Note::new(
            recipient,
            NoteValue::from_raw(5),
            rho,
            NoteVersion::V3,
            &mut rng,
        );
        let memo = [0u8; 512];
        let cv_net = ValueCommitment::derive(ValueSum::from_raw(5), ValueCommitTrapdoor::zero());
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        let encryptor = IronwoodNoteEncryption::new(Some(ovk.clone()), note, memo);
        let epk = IronwoodDomain::epk_bytes(encryptor.epk());
        let enc_ct = encryptor.encrypt_note_plaintext();
        let out_ct = encryptor.encrypt_outgoing_plaintext(&cv_net, &cmx, &mut rng);
        let encrypted_note = TransmittedNoteCiphertext {
            epk_bytes: epk.0,
            enc_ciphertext: enc_ct,
            out_ciphertext: out_ct,
        };
        let action = Action::from_parts(
            nf_old,
            redpallas::VerificationKey::dummy(),
            cmx,
            encrypted_note,
            cv_net.clone(),
            (),
        )
        .unwrap();
        let ock = prf_ock_orchard(&ovk, &cv_net, &cmx.to_bytes(), &epk);
        FullAction {
            fvk,
            action,
            note,
            recipient,
            memo,
            ovk,
            cv_net,
            out_ct,
            epk,
            ock,
        }
    }

    #[test]
    fn sinsemilla_call_count_per_action_baseline() {
        // Stock per-action verification path (no dedup), measured with the same
        // instrumented Sinsemilla in the same tree, to establish the BEFORE count.
        // Must be run isolated (`--test-threads=1`, single filter) because the
        // counters are process-global atomics.
        use zcash_note_encryption::{
            try_output_recovery_with_ock, try_output_recovery_with_pkd_esk,
        };
        fn dump(label: &str, before: sinsemilla::SinsemillaCounters) {
            let a = sinsemilla::counters_snapshot();
            std::println!(
                "COUNT[{}]: htp_calls={} htp_chunks={} commit={} short_commit={} commitdomain_new={} hashdomain_new={}",
                label,
                a.htp_calls - before.htp_calls,
                a.htp_chunks - before.htp_chunks,
                a.commit_calls - before.commit_calls,
                a.shortcommit_calls - before.shortcommit_calls,
                a.commitdomain_new - before.commitdomain_new,
                a.hashdomain_new - before.hashdomain_new,
            );
        }
        let fa = full_v3_action();
        let fvk = &fa.fvk;
        let action = &fa.action;
        let recipient = fa.recipient;
        let rho = Rho::from_nf_old(*action.nullifier());
        let value = fa.note.value();
        let rseed = fa.note.rseed().clone();
        let cmx = *action.cmx();
        let domain = IronwoodDomain::for_action(action);
        let pk_d = IronwoodDomain::get_pk_d(&fa.note);
        let esk = IronwoodDomain::derive_esk(&fa.note).unwrap();
        let out_ct = fa.out_ct;

        let before_action = sinsemilla::counters_snapshot();
        {
            // verify_nullifier
            let ns =
                Note::from_parts(recipient, value, rho, rseed.clone(), NoteVersion::V3).unwrap();
            let _ = fvk.scope_for_address(&ns.recipient());
            let _ = ns.nullifier(fvk);
            // verify_note_commitment
            let n0 =
                Note::from_parts(recipient, value, rho, rseed.clone(), NoteVersion::V3).unwrap();
            let _ = ExtractedNoteCommitment::from(n0.commitment()) == cmx;
            // scope_for_address(recipient) in verify_bundle
            let _ = fvk.scope_for_address(&recipient);
            // verify_encryption (STOCK engine wiring: recover + Note::eq checks).
            // The stock engine rebuilds the note, recovers via pkd_esk, then
            // compares the recovered note/tuple for equality with `==`. Because
            // `impl PartialEq for Note` recomputes BOTH commitments (note.rs),
            // each `==` is 2 NoteCommits — the omission that made the prior
            // baseline read 13 instead of the true ~19 (Fable review #7).
            let en =
                Note::from_parts(recipient, value, rho, rseed.clone(), NoteVersion::V3).unwrap();
            let recovered = try_output_recovery_with_pkd_esk(&domain, pk_d, esk, action).unwrap();
            let _ = recovered.0 == en && recovered.1 == en.recipient(); // Note::eq: 2 NoteCommit
            let r_ock = try_output_recovery_with_ock(&domain, &fa.ock, action, &out_ct);
            let _ = r_ock.is_some_and(|r| r == recovered); // tuple eq -> Note::eq: 2 NoteCommit
            let r_ovk = try_output_recovery_with_ovk(&domain, &fa.ovk, action, &fa.cv_net, &out_ct);
            let _ = r_ovk.is_some_and(|r| r == recovered); // tuple eq -> Note::eq: 2 NoteCommit
        }
        dump(
            "WHOLE_PER_ACTION_BASELINE (stock path, engine eq checks)",
            before_action,
        );
    }

    #[test]
    fn sinsemilla_call_count_per_action_deduped() {
        use super::{
            recover_output_bound_with_ock, recover_output_bound_with_ovk,
            recover_output_bound_with_pkd_esk,
        };

        fn dump(label: &str, before: sinsemilla::SinsemillaCounters) {
            let a = sinsemilla::counters_snapshot();
            std::println!(
                "COUNT[{}]: htp_calls={} htp_chunks={} commit={} short_commit={} commitdomain_new={} hashdomain_new={}",
                label,
                a.htp_calls - before.htp_calls,
                a.htp_chunks - before.htp_chunks,
                a.commit_calls - before.commit_calls,
                a.shortcommit_calls - before.shortcommit_calls,
                a.commitdomain_new - before.commitdomain_new,
                a.hashdomain_new - before.hashdomain_new,
            );
        }

        let fa = full_v3_action();
        let fvk = &fa.fvk;
        let action = &fa.action;
        let recipient = fa.recipient;
        let rho = Rho::from_nf_old(*action.nullifier());
        let value = fa.note.value();
        let rseed = fa.note.rseed().clone();
        let cmx_expected = *action.cmx();

        let domain = IronwoodDomain::for_action(action);
        let pk_d = IronwoodDomain::get_pk_d(&fa.note);
        let esk = IronwoodDomain::derive_esk(&fa.note).unwrap();
        let out_ct = fa.out_ct;

        // ---- SESSION PHASE (amortized, once per bundle): cache ivk ----
        let session_before = sinsemilla::counters_snapshot();
        let classifier = fvk.scope_classifier();
        dump(
            "SESSION scope_classifier (cached ivk, once per bundle)",
            session_before,
        );

        // ---- DEDUPED WHOLE PER ACTION ----
        let before_action = sinsemilla::counters_snapshot();
        {
            // verify_nullifier: build spend note + cm_old ONCE, reuse for nullifier.
            let b = sinsemilla::counters_snapshot();
            let (spend_note, cm_old) = Note::from_parts_with_commitment(
                recipient,
                value,
                rho,
                rseed.clone(),
                NoteVersion::V3,
            )
            .unwrap();
            dump("  step verify_nullifier from_parts_with_commitment", b);
            let b = sinsemilla::counters_snapshot();
            let _scope = classifier.scope_for_address(&spend_note.recipient()); // cached ivk: 0 Sinsemilla
            dump("  step scope_for_address(spend)", b);
            let b = sinsemilla::counters_snapshot();
            let _nf = spend_note.nullifier_with_commitment(fvk, &cm_old); // reuse cm_old: 0 Sinsemilla
            dump("  step nullifier_with_commitment", b);

            // verify_note_commitment: build output note + cmx ONCE, compare.
            let b = sinsemilla::counters_snapshot();
            let (out_note, cm) = Note::from_parts_with_commitment(
                recipient,
                value,
                rho,
                rseed.clone(),
                NoteVersion::V3,
            )
            .unwrap();
            dump(
                "  step verify_note_commitment from_parts_with_commitment",
                b,
            );
            let cmx = ExtractedNoteCommitment::from(cm);
            assert_eq!(cmx, cmx_expected, "verify_note_commitment must still hold");

            // scope_for_address(recipient) in verify_bundle: cached ivk, 0 Sinsemilla.
            let _rscope = classifier.scope_for_address(&recipient);

            // verify_encryption: device-local recovery bound to `out_note`, 0 Sinsemilla.
            let b = sinsemilla::counters_snapshot();
            let m1 = recover_output_bound_with_pkd_esk(&domain, pk_d, esk, action, &out_note)
                .expect("pkd_esk recovery must accept");
            dump("  step recover pkd_esk", b);
            let b = sinsemilla::counters_snapshot();
            let m2 = recover_output_bound_with_ock(&domain, &fa.ock, action, &out_ct, &out_note)
                .expect("ock recovery must accept");
            dump("  step recover ock", b);
            let b = sinsemilla::counters_snapshot();
            let m3 = recover_output_bound_with_ovk(
                &domain, &fa.ovk, action, &fa.cv_net, &out_ct, &out_note,
            )
            .expect("ovk recovery must accept");
            dump("  step recover ovk", b);
            assert_eq!(m1, fa.memo);
            assert_eq!(m2, fa.memo);
            assert_eq!(m3, fa.memo);
        }
        dump(
            "WHOLE_PER_ACTION_DEDUPED (2 NoteCommit: cm_old + cmx; 0 CommitIvk)",
            before_action,
        );

        // Hard assertions: exactly 2 hash_to_point evals, 0 short-commit per action.
        let after = sinsemilla::counters_snapshot();
        let htp = after.htp_calls - before_action.htp_calls;
        let sc = after.shortcommit_calls - before_action.shortcommit_calls;
        let cdn = after.commitdomain_new - before_action.commitdomain_new;
        assert_eq!(htp, 2, "expected exactly 2 hash_to_point evals per action");
        assert_eq!(sc, 0, "expected 0 CommitIvk short-commits per action");
        assert_eq!(
            cdn, 0,
            "CommitDomain must be cached, not rebuilt per action"
        );
    }

    #[test]
    fn deduped_recovery_accepts_valid_and_matches_upstream() {
        use super::recover_output_bound_with_ovk;
        let fa = full_v3_action();
        let domain = IronwoodDomain::for_action(&fa.action);

        // Deduped path accepts and returns the correct memo.
        let memo = recover_output_bound_with_ovk(
            &domain, &fa.ovk, &fa.action, &fa.cv_net, &fa.out_ct, &fa.note,
        );
        assert_eq!(memo, Some(fa.memo));

        // Cross-check: stock upstream recovery agrees (same note recovered).
        let upstream =
            try_output_recovery_with_ovk(&domain, &fa.ovk, &fa.action, &fa.cv_net, &fa.out_ct);
        let (u_note, u_to, u_memo) = upstream.expect("stock recovery accepts valid action");
        assert_eq!(u_note, fa.note);
        assert_eq!(u_to, fa.recipient);
        assert_eq!(u_memo, fa.memo);
    }

    #[test]
    fn deduped_recovery_rejects_tampered_ciphertext() {
        use super::recover_output_bound_with_pkd_esk;
        let mut fa = full_v3_action();
        let domain = IronwoodDomain::for_action(&fa.action);
        let pk_d = IronwoodDomain::get_pk_d(&fa.note);
        let esk = IronwoodDomain::derive_esk(&fa.note).unwrap();

        // Flip a byte in the note ciphertext -> AEAD authentication fails.
        let mut tampered = fa.action.encrypted_note().clone();
        tampered.enc_ciphertext[0] ^= 0x01;
        fa.action = Action::from_parts(
            *fa.action.nullifier(),
            redpallas::VerificationKey::dummy(),
            *fa.action.cmx(),
            tampered,
            fa.cv_net.clone(),
            (),
        )
        .unwrap();

        let out = recover_output_bound_with_pkd_esk(&domain, pk_d, esk, &fa.action, &fa.note);
        assert_eq!(out, None, "tampered ciphertext must be rejected");
    }

    #[test]
    fn deduped_recovery_rejects_wrong_expected_note_fields() {
        use super::recover_output_bound_with_ovk;
        let fa = full_v3_action();
        let domain = IronwoodDomain::for_action(&fa.action);

        // Bind against a note with a DIFFERENT value than the ciphertext encodes.
        // The plaintext decrypts fine, but the field comparison must reject it,
        // proving the binding to the already-validated note is enforced (this is
        // the security property that replaces the cmstar recomputation).
        let rho = Rho::from_nf_old(*fa.action.nullifier());
        let wrong_note = Note::from_parts(
            fa.recipient,
            NoteValue::from_raw(999),
            rho,
            fa.note.rseed().clone(),
            NoteVersion::V3,
        )
        .unwrap();

        let out = recover_output_bound_with_ovk(
            &domain,
            &fa.ovk,
            &fa.action,
            &fa.cv_net,
            &fa.out_ct,
            &wrong_note,
        );
        assert_eq!(
            out, None,
            "recovery bound to a mismatched note must be rejected"
        );
    }

    #[test]
    fn deduped_verify_note_commitment_rejects_wrong_cmx() {
        // The cmx binding (verify_note_commitment) is unchanged and is the sole
        // place cmx is derived. A tampered action cmx must still be rejected.
        let fa = full_v3_action();
        let rho = Rho::from_nf_old(*fa.action.nullifier());
        let (_note, cm) = Note::from_parts_with_commitment(
            fa.recipient,
            fa.note.value(),
            rho,
            fa.note.rseed().clone(),
            NoteVersion::V3,
        )
        .unwrap();
        let cmx = ExtractedNoteCommitment::from(cm);

        // Correct action cmx accepts.
        assert_eq!(cmx, *fa.action.cmx());

        // A different cmx (simulating a tampered action) is rejected.
        let wrong_note = Note::new(
            fa.recipient,
            NoteValue::from_raw(7),
            rho,
            NoteVersion::V3,
            &mut OsRng,
        );
        let wrong_cmx = ExtractedNoteCommitment::from(wrong_note.commitment());
        assert_ne!(cmx, wrong_cmx, "tampered cmx must not match");
    }

    #[test]
    fn deduped_recovery_rejects_wrong_rho_domain() {
        // MUST-FIX #1: `rho` is the single NoteCommit input not present in the
        // plaintext. The recovery variant must reject an `expected` note whose
        // rho differs from the domain's (i.e. the action's), even though every
        // in-plaintext field (diversifier, value, rseed, pk_d, version) matches.
        // This is the case that made the pre-fix API strictly weaker than the
        // upstream `cmstar` recomputation it replaced.
        use super::recover_output_bound_with_pkd_esk;
        let fa = full_v3_action();
        let domain = IronwoodDomain::for_action(&fa.action);
        let pk_d = IronwoodDomain::get_pk_d(&fa.note);
        let esk = IronwoodDomain::derive_esk(&fa.note).unwrap();

        // Same fields, but a DIFFERENT rho than the action/domain.
        let other_rho = Rho::from_nf_old(Nullifier::dummy(&mut OsRng));
        let wrong_rho_note = Note::from_parts(
            fa.recipient,
            fa.note.value(),
            other_rho,
            fa.note.rseed().clone(),
            NoteVersion::V3,
        )
        .unwrap();
        assert_ne!(domain.rho, wrong_rho_note.rho());

        let out =
            recover_output_bound_with_pkd_esk(&domain, pk_d, esk, &fa.action, &wrong_rho_note);
        assert_eq!(
            out, None,
            "recovery must reject a note whose rho != domain.rho"
        );
    }

    #[test]
    fn deduped_recovery_rejects_wrong_domain_version_policy() {
        // MUST-FIX #1: the accepted note version is a property of the *domain*
        // policy, not of `expected`. Recovering a V3 action's output under the
        // V2 `OrchardDomain` must reject (the `0x03` lead byte is not accepted by
        // the V2 policy), matching upstream `try_output_recovery_*` under that
        // domain.
        use super::recover_output_bound_with_pkd_esk;
        let fa = full_v3_action();
        let orchard_domain = OrchardDomain::for_action(&fa.action);
        let pk_d = IronwoodDomain::get_pk_d(&fa.note);
        let esk = IronwoodDomain::derive_esk(&fa.note).unwrap();

        let out =
            recover_output_bound_with_pkd_esk(&orchard_domain, pk_d, esk, &fa.action, &fa.note);
        assert_eq!(
            out, None,
            "V2 domain policy must reject a V3 note plaintext (lead byte 0x03)"
        );
    }

    /// Encrypts a compact output of the domain's note plaintext version to
    /// `recipient`, using a fresh ephemeral key.
    fn encrypted_compact_action<V: DomainVersion>(
        rng: &mut OsRng,
        recipient: Address,
    ) -> CompactAction {
        let nf_old = Nullifier::dummy(rng);
        let rho = Rho::from_nf_old(nf_old);
        let note = Note::new(
            recipient,
            NoteValue::from_raw(42),
            rho,
            V::NOTE_VERSION,
            rng,
        );
        let encryptor = NoteEncryption::<NoteEncryptionDomain<V>>::new(None, note, [0u8; 512]);
        let ephemeral_key = NoteEncryptionDomain::<V>::epk_bytes(encryptor.epk());
        let enc_ciphertext = encryptor.encrypt_note_plaintext();
        CompactAction::from_parts(
            nf_old,
            ExtractedNoteCommitment::from(note.commitment()),
            ephemeral_key,
            enc_ciphertext.as_ref()[..52].try_into().unwrap(),
        )
    }

    /// The batched trial-decryption pipeline (GLV-window preparation and
    /// per-batch scalar decomposition) must produce exactly the per-item
    /// results, over hits on multiple viewing keys, misses, and an
    /// undecodable ephemeral key.
    fn check_batched_compact_decryption_matches_per_item<V: DomainVersion>() {
        let mut rng = OsRng;

        // Two accounts with external and internal scope each — the wallet
        // shape batched trial decryption runs with — plus a foreign account
        // whose outputs must not decrypt.
        let our_fvk = FullViewingKey::from(&SpendingKey::random(&mut rng));
        let other_fvk = FullViewingKey::from(&SpendingKey::random(&mut rng));
        let foreign_fvk = FullViewingKey::from(&SpendingKey::random(&mut rng));
        let ivks: Vec<PreparedIncomingViewingKey> = [
            (&our_fvk, Scope::External),
            (&our_fvk, Scope::Internal),
            (&other_fvk, Scope::External),
            (&other_fvk, Scope::Internal),
        ]
        .into_iter()
        .map(|(fvk, scope)| PreparedIncomingViewingKey::new(&fvk.to_ivk(scope)))
        .collect();

        let mut actions = vec![
            encrypted_compact_action::<V>(&mut rng, our_fvk.address_at(0u32, Scope::External)),
            encrypted_compact_action::<V>(&mut rng, our_fvk.address_at(0u32, Scope::Internal)),
            encrypted_compact_action::<V>(&mut rng, other_fvk.address_at(0u32, Scope::External)),
        ];
        for i in 0..5u32 {
            actions.push(encrypted_compact_action::<V>(
                &mut rng,
                foreign_fvk.address_at(i, Scope::External),
            ));
        }
        // An ephemeral key that decodes to the identity is rejected during
        // preparation; its lane must pass through as `None`.
        actions.push(CompactAction::from_parts(
            Nullifier::dummy(&mut rng),
            actions[0].cmx(),
            EphemeralKeyBytes([0u8; 32]),
            [0u8; 52],
        ));

        let items: Vec<(NoteEncryptionDomain<V>, CompactAction)> = actions
            .iter()
            .map(|a| (NoteEncryptionDomain::<V>::for_compact_action(a), a.clone()))
            .collect();
        let batched = batch::try_compact_note_decryption(&ivks, &items);

        let per_item: Vec<Option<((Note, Address), usize)>> = actions
            .iter()
            .map(|a| {
                let domain = NoteEncryptionDomain::<V>::for_compact_action(a);
                ivks.iter().enumerate().find_map(|(i, ivk)| {
                    try_compact_note_decryption(&domain, ivk, a).map(|r| (r, i))
                })
            })
            .collect();

        assert_eq!(batched, per_item);

        // The interesting lanes actually decrypted (guards against both
        // paths failing identically).
        assert_eq!(batched[0].as_ref().map(|(_, i)| *i), Some(0));
        assert_eq!(batched[1].as_ref().map(|(_, i)| *i), Some(1));
        assert_eq!(batched[2].as_ref().map(|(_, i)| *i), Some(2));
        assert!(batched[3..].iter().all(Option::is_none));
    }

    #[test]
    fn batched_compact_decryption_matches_per_item_orchard() {
        check_batched_compact_decryption_matches_per_item::<OrchardVersion>();
    }

    #[test]
    fn batched_compact_decryption_matches_per_item_ironwood() {
        check_batched_compact_decryption_matches_per_item::<IronwoodVersion>();
    }

    /// The batched agreement must produce byte-identical shared secrets to
    /// the per-item path, for both preparation routes (batch-built GLV
    /// windows and individually-built wNAF tables), on hit and miss lanes
    /// alike.
    #[test]
    fn batched_agreement_matches_per_item() {
        let mut rng = OsRng;

        let our_fvk = FullViewingKey::from(&SpendingKey::random(&mut rng));
        let foreign_fvk = FullViewingKey::from(&SpendingKey::random(&mut rng));
        let ivks: Vec<PreparedIncomingViewingKey> = [
            (&our_fvk, Scope::External),
            (&our_fvk, Scope::Internal),
            (&foreign_fvk, Scope::External),
        ]
        .into_iter()
        .map(|(fvk, scope)| PreparedIncomingViewingKey::new(&fvk.to_ivk(scope)))
        .collect();

        // Real ephemeral keys from real encryptions, plus an undecodable lane.
        let mut keys: Vec<EphemeralKeyBytes> = (0..12u32)
            .map(|i| {
                encrypted_compact_action::<OrchardVersion>(
                    &mut rng,
                    our_fvk.address_at(i, Scope::External),
                )
                .ephemeral_key
            })
            .collect();
        keys.push(EphemeralKeyBytes([0u8; 32]));

        let batch_prepared = <OrchardDomain as BatchDomain>::batch_epk(keys.iter().cloned());
        // The undecodable lane passes through preparation as `None`.
        assert!(batch_prepared.last().unwrap().0.is_none());
        assert!(batch_prepared[..12].iter().all(|(p, _)| p.is_some()));

        let wnaf_prepared: Vec<Option<crate::keys::PreparedEphemeralPublicKey>> = keys
            .iter()
            .map(|key| OrchardDomain::epk(key).map(OrchardDomain::prepare_epk))
            .collect();

        for ivk in &ivks {
            let expected: Vec<Option<[u8; 32]>> = keys
                .iter()
                .map(|key| {
                    OrchardDomain::epk(key)
                        .map(OrchardDomain::prepare_epk)
                        .map(|epk| OrchardDomain::ka_agree_dec(ivk, &epk).to_bytes())
                })
                .collect();

            let batched: Vec<Option<[u8; 32]>> =
                <OrchardDomain as BatchDomain>::batch_ka_agree_dec(
                    ivk,
                    batch_prepared.iter().map(|(p, _)| p.as_ref()),
                )
                .into_iter()
                .map(|s| s.map(|s| s.to_bytes()))
                .collect();
            assert_eq!(batched, expected);

            // The batched agreement's fallback arm (individually-prepared
            // inputs) must also match.
            let batched_wnaf: Vec<Option<[u8; 32]>> =
                <OrchardDomain as BatchDomain>::batch_ka_agree_dec(
                    ivk,
                    wnaf_prepared.iter().map(|p| p.as_ref()),
                )
                .into_iter()
                .map(|s| s.map(|s| s.to_bytes()))
                .collect();
            assert_eq!(batched_wnaf, expected);
        }
    }
}
