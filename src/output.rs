use std::net::{ToSocketAddrs, UdpSocket};

use rosc::{OscMessage, OscPacket, OscType, encoder};
use thiserror::Error;

use crate::calibration::{ShapeWeight, eye::EyeShape, face::FaceShape};

pub struct OscTransport {
    socket: UdpSocket,
    destination: std::net::SocketAddr,
}

#[derive(Clone, Debug, Error)]
pub enum TransportError {
    #[error("failed to bind UDP socket")]
    Bind,
    #[error("failed to resolve destination address")]
    Resolve,
}

impl OscTransport {
    pub fn udp(destination: impl ToSocketAddrs) -> Result<Self, TransportError> {
        Ok(Self {
            socket: UdpSocket::bind("0.0.0.0:0").map_err(|_| TransportError::Bind)?,

            destination: destination
                .to_socket_addrs()
                .map_err(|_| TransportError::Resolve)?
                .next()
                .ok_or(TransportError::Resolve)?,
        })
    }

    pub(crate) fn send(&mut self, msg: OscMessage) {
        let msg = OscPacket::Message(msg);

        if let Ok(buf) = encoder::encode(&msg) {
            let _ = self.socket.send_to(&buf, &self.destination);
        }
    }

    pub fn flush(&mut self) {
        // No-op for now
    }
}

pub struct BabbleEmitter {
    // TODO
}

impl BabbleEmitter {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_face(
        &mut self,
        weights: &[ShapeWeight<FaceShape>],
        transport: &mut OscTransport,
    ) {
        for weight in weights {
            let msg = OscMessage {
                addr: weight.shape.to_babble().to_string(),
                args: vec![OscType::Float(weight.value)],
            };

            transport.send(msg);
        }
    }

    pub fn process_eyes(
        &mut self,
        weights: &[ShapeWeight<EyeShape>],
        transport: &mut OscTransport,
    ) {
        let _ = (weights, transport);
        todo!()
    }
}

pub struct EtvrEmitter {
    // TODO
}

impl EtvrEmitter {
    pub fn new() -> Self {
        todo!()
    }

    pub fn process_eyes(
        &mut self,
        weights: &[ShapeWeight<EyeShape>],
        transport: &mut OscTransport,
    ) {
        let _ = (weights, transport);
        todo!()
    }
}

pub struct NativeEmitter {
    // TODO
}

impl NativeEmitter {
    pub fn new() -> Self {
        todo!()
    }

    pub fn process_eyes(
        &mut self,
        weights: &[ShapeWeight<EyeShape>],
        transport: &mut OscTransport,
    ) {
        let _ = (weights, transport);
        todo!()
    }
}
