use std::net::ToSocketAddrs;

use crate::{
    calibration::{EyeShape, FaceShape, Weights},
    output::{BabbleEmitter, EtvrEmitter, OscTransport, TransportError},
};

pub struct Output {
    pub transport: OscTransport,
    pub babble: BabbleEmitter,
    pub etvr: EtvrEmitter,
}

impl Output {
    pub fn new(destination: impl ToSocketAddrs) -> Result<Self, TransportError> {
        Ok(Self {
            transport: OscTransport::udp(destination)?,
            babble: BabbleEmitter::new(),
            etvr: EtvrEmitter::new(),
        })
    }

    pub fn set_destination(
        &mut self,
        destination: impl ToSocketAddrs,
    ) -> Result<(), TransportError> {
        self.transport = OscTransport::udp(destination)?;
        Ok(())
    }

    pub fn send_face(&mut self, weights: Weights<'_, FaceShape>) {
        self.babble.process_face(weights, &mut self.transport);
    }

    pub fn send_eyes(&mut self, weights: Weights<'_, EyeShape>) {
        self.babble.process_eyes(weights, &mut self.transport);
        self.etvr.process_eyes(weights, &mut self.transport);
    }

    pub fn flush(&mut self) -> Result<(), TransportError> {
        self.transport.flush()
    }
}
